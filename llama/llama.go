// Package llama provides Go bindings to llama.cpp via CGO
//
// This is the bridge between Go code and C/C++/CUDA inference engine.
// All actual model inference happens through these CGO calls.
//
// Key components:
// - LoadModelFromFile(): Loads GGUF model file into memory
// - Context.Decode(): Runs inference (GPU/CPU matrix operations)
// - SamplingContext: Selects next token from logits
// - Batch: Groups tokens for efficient parallel processing
//
// For Tesla K80 (compute 3.7):
// - CUDA kernels compiled with PTX (JIT at runtime)
// - Model layers distributed across CPU/GPU based on num_gpu_layers
// - KV cache allocated on GPU VRAM (12GB available)
package llama

/*
#cgo CFLAGS: -std=c11
#cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
#cgo CXXFLAGS: -std=c++17
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/include
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/common
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/vendor
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/tools/mtmd
#cgo CPPFLAGS: -I${SRCDIR}/llama.cpp/src
#cgo CPPFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include

#include <stdlib.h>
#include "ggml.h"       // GGML tensor library (CPU/GPU operations)
#include "llama.h"      // llama.cpp model loading and inference
#include "mtmd.h"       // Multi-turn multi-document support
#include "mtmd-helper.h"
#include "gguf.h"       // GGUF file format parsing
#include "sampling_ext.h" // Token sampling (temperature, top_p, etc.)

extern bool llamaProgressCallback(float progress, void *user_data);
extern void llamaLog(int level, char* text, void* user_data);
*/
import "C"

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"runtime/cgo"
	"slices"
	"strings"
	"sync"
	"time"
	"unsafe"

	_ "github.com/ollama/ollama/llama/llama.cpp/common"
	_ "github.com/ollama/ollama/llama/llama.cpp/src"
	_ "github.com/ollama/ollama/llama/llama.cpp/tools/mtmd"
	"github.com/ollama/ollama/ml"
	ggml "github.com/ollama/ollama/ml/backend/ggml/ggml/src"
)

func init() {
	C.llama_log_set(C.ggml_log_callback(C.llamaLog), nil)
}

//export llamaLog
func llamaLog(level C.int, text *C.char, _ unsafe.Pointer) {
	// slog levels zeros INFO and are multiples of 4
	if slog.Default().Enabled(context.TODO(), slog.Level(int(level-C.GGML_LOG_LEVEL_INFO)*4)) {
		fmt.Fprint(os.Stderr, C.GoString(text))
	}
}

// BackendInit initializes the llama.cpp backend
//
// This must be called once before loading any models.
// It initializes:
// - CUDA backend (if GPUs available)
// - CPU backend
// - Memory allocators
// - Threading infrastructure
//
// For Tesla K80 (compute 3.7):
// - Detects CUDA device
// - Verifies compute capability support
// - Initializes cuBLAS for matrix operations
func BackendInit() {
	ggml.OnceLoad() // Load GGML shared library
	C.llama_backend_init() // Initialize llama.cpp backend
}

func EnumerateGPUs() []ml.DeviceID {
	var ids []ml.DeviceID

	for i := range C.ggml_backend_dev_count() {
		device := C.ggml_backend_dev_get(i)

		switch C.ggml_backend_dev_type(device) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			var props C.struct_ggml_backend_dev_props
			C.ggml_backend_dev_get_props(device, &props)
			ids = append(ids, ml.DeviceID{
				ID:      C.GoString(props.id),
				Library: C.GoString(props.library),
			})
		}
	}

	return ids
}

func GetModelArch(modelPath string) (string, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))

	gguf_ctx := C.gguf_init_from_file(mp, C.struct_gguf_init_params{no_alloc: true, ctx: (**C.struct_ggml_context)(C.NULL)})
	if gguf_ctx == nil {
		return "", errors.New("unable to load model file")
	}
	defer C.gguf_free(gguf_ctx)

	key := C.CString("general.architecture")
	defer C.free(unsafe.Pointer(key))
	arch_index := C.gguf_find_key(gguf_ctx, key)
	if int(arch_index) < 0 {
		return "", errors.New("unknown model architecture")
	}

	arch := C.gguf_get_val_str(gguf_ctx, arch_index)

	return C.GoString(arch), nil
}

type ContextParams struct {
	c C.struct_llama_context_params
}

func NewContextParams(numCtx int, batchSize int, numSeqMax int, threads int, flashAttention bool, kvCacheType string) ContextParams {
	params := C.llama_context_default_params()
	params.n_ctx = C.uint(numCtx)
	params.n_batch = C.uint(batchSize)
	params.n_seq_max = C.uint(numSeqMax)
	params.n_threads = C.int(threads)
	params.n_threads_batch = params.n_threads
	params.embeddings = C.bool(true)
	if flashAttention {
		params.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	} else {
		params.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	}
	params.type_k = kvCacheTypeFromStr(strings.ToLower(kvCacheType))
	params.type_v = kvCacheTypeFromStr(strings.ToLower(kvCacheType))

	// Size the SWA (sliding-window) KV cache to the window, not the full context.
	// llama.cpp defaults swa_full=true (a conservative correctness choice for context
	// reprocessing); for autoregressive generation the windowed cache is correct, and on
	// the memory-constrained K80 the full-size cache is wasteful (e.g. gemma4:12b @16k: a
	// 40-layer SWA cache at full context is ~5.1 GiB vs ~0.6 GiB windowed).
	params.swa_full = C.bool(false)

	return ContextParams{c: params}
}

// kvCacheTypeFromStr converts a string cache type to the corresponding GGML type value
func kvCacheTypeFromStr(s string) C.enum_ggml_type {
	if s == "" {
		return C.GGML_TYPE_F16
	}

	switch s {
	case "q8_0":
		return C.GGML_TYPE_Q8_0
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
		return C.GGML_TYPE_F16
	}
}

// Context represents an active inference context
//
// This wraps llama.cpp's llama_context which holds:
// - Model pointer
// - KV cache (key-value cache for attention)
// - Thread pool for CPU operations
// - RNG state for sampling
//
// Each Context can handle multiple parallel sequences (controlled by numParallel)
type Context struct {
	c          *C.struct_llama_context // C pointer to llama_context
	numThreads int                     // Number of CPU threads for inference
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

// Decode runs one inference step on a batch of tokens
//
// *** THIS IS WHERE ACTUAL INFERENCE HAPPENS ***
//
// For each token in the batch, this:
// 1. Retrieves token embeddings from model
// 2. Runs through transformer layers:
//    - Attention (uses KV cache)
//    - Feed-forward network
//    - Layer normalization
// 3. Stores KV states in cache for future tokens
// 4. Produces output logits (probabilities for next token)
//
// GPU execution (Tesla K80, compute 3.7):
// - Matrix multiplications via cuBLAS
// - Attention via CUDA kernels
// - LayerNorm/RoPE/Softmax via CUDA kernels
// - Data transferred between CPU/GPU as needed
//
// Returns:
// - nil: success
// - ErrKvCacheFull: no space in KV cache (increase num_ctx or reduce batch size)
// - error: fatal error during inference
func (c *Context) Decode(batch *Batch) error {
	// Call C function: int llama_decode(llama_context*, llama_batch)
	// This executes the actual neural network forward pass
	//
	// Return codes:
	//   0 - success
	//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
	// < 0 - error
	code := int(C.llama_decode(c.c, batch.c))

	if code < 0 {
		return fmt.Errorf("llama_decode failed with code %d", code)
	}

	if code > 0 {
		return ErrKvCacheFull
	}

	return nil
}

func (c *Context) Model() *Model {
	return &Model{c: C.llama_get_model(c.c)}
}

func (c *Context) KvCacheSeqAdd(seqId int, p0 int, p1 int, delta int) {
	C.llama_memory_seq_add(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1), C.int(delta))
}

func (c *Context) KvCacheSeqRm(seqId int, p0 int, p1 int) bool {
	return bool(C.llama_memory_seq_rm(C.llama_get_memory(c.c), C.int(seqId), C.int(p0), C.int(p1)))
}

func (c *Context) KvCacheSeqCp(srcSeqId int, dstSeqId int, p0 int, p1 int) {
	C.llama_memory_seq_cp(C.llama_get_memory(c.c), C.int(srcSeqId), C.int(dstSeqId), C.int(p0), C.int(p1))
}

func (c *Context) KvCacheClear() {
	C.llama_memory_clear(C.llama_get_memory(c.c), true)
}

func (c *Context) KvCacheCanShift() bool {
	return bool(C.llama_memory_can_shift(C.llama_get_memory(c.c)))
}

// Get the embeddings for a sequence id
func (c *Context) GetEmbeddingsSeq(seqId int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_seq(c.c, C.int(seqId)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

func (c *Context) GetEmbeddingsIth(i int) []float32 {
	e := unsafe.Pointer(C.llama_get_embeddings_ith(c.c, C.int32_t(i)))
	if e == nil {
		return nil
	}

	embeddings := make([]float32, c.Model().NEmbd())
	_ = copy(embeddings, unsafe.Slice((*float32)(e), c.Model().NEmbd()))
	return embeddings
}

type ModelParams struct {
	NumGpuLayers int
	MainGpu      int
	UseMmap      bool
	TensorSplit  []float32
	Progress     func(float32)
	VocabOnly    bool
}

//export llamaProgressCallback
func llamaProgressCallback(progress C.float, userData unsafe.Pointer) C.bool {
	handle := *(*cgo.Handle)(userData)
	callback := handle.Value().(func(float32))
	callback(float32(progress))
	return true
}

// LoadModelFromFile loads a GGUF model file into memory
//
// *** THIS IS THE CORE MODEL LOADING FUNCTION ***
//
// This reads the GGUF file and loads model weights into memory.
// The process:
// 1. Parse GGUF file headers (metadata, architecture, tensors)
// 2. Memory-map (mmap) or read model weights
// 3. Distribute layers across devices based on NumGpuLayers:
//    - First NumGpuLayers transformer layers → GPU
//    - Remaining layers → CPU
//    - Embeddings and output layer handling varies
// 4. Allocate device buffers for tensors
//
// For Tesla K80 (compute 3.7) with 12GB VRAM:
// - Example: gemma3-2b with Q4_0 quantization
//   - Full model ~1.5GB, can fit entirely on GPU (NumGpuLayers=99)
// - Example: llama3-8b with Q4_0 quantization
//   - Full model ~4.5GB, can fit entirely on GPU
// - Example: llama3-70b with Q4_0 quantization
//   - Full model ~40GB, needs CPU offload (NumGpuLayers=20-30)
//
// Parameters:
// - modelPath: Path to GGUF file (e.g., ~/.ollama/models/blobs/sha256-abc123...)
// - params.NumGpuLayers: How many transformer layers to put on GPU
// - params.MainGpu: Which GPU to use (if multiple)
// - params.TensorSplit: How to split layers across multiple GPUs
// - params.UseMmap: Whether to use memory mapping (faster load, less RAM)
//
// Returns:
// - *Model: Loaded model ready for inference
// - error: If file not found, incompatible format, or out of memory
func LoadModelFromFile(modelPath string, params ModelParams) (*Model, error) {
	loadStart := time.Now()
	slog.Info("LoadModelFromFile: starting", "path", modelPath, "num_gpu_layers", params.NumGpuLayers, "use_mmap", params.UseMmap)

	// Initialize C parameters structure
	cparams := C.llama_model_default_params()
	cparams.n_gpu_layers = C.int(params.NumGpuLayers) // Layers to offload to GPU
	cparams.main_gpu = C.int32_t(params.MainGpu)       // Primary GPU device ID
	cparams.use_mmap = C.bool(params.UseMmap)          // Memory-map file (faster)
	cparams.vocab_only = C.bool(params.VocabOnly)      // Load vocabulary only

	// Multi-GPU tensor split (for systems with multiple GPUs)
	// Defines proportion of model to put on each GPU
	if len(params.TensorSplit) > 0 {
		tensorSplitData := &params.TensorSplit[0]

		var tensorSplitPin runtime.Pinner
		tensorSplitPin.Pin(tensorSplitData)
		defer tensorSplitPin.Unpin()

		cparams.tensor_split = (*C.float)(unsafe.Pointer(tensorSplitData))
	}

	// Progress callback (reports loading progress percentage)
	if params.Progress != nil {
		handle := cgo.NewHandle(params.Progress)
		defer handle.Delete()

		var handlePin runtime.Pinner
		handlePin.Pin(&handle)
		defer handlePin.Unpin()

		cparams.progress_callback = C.llama_progress_callback(C.llamaProgressCallback)
		cparams.progress_callback_user_data = unsafe.Pointer(&handle)
	}

	// *** CALL C FUNCTION TO LOAD MODEL ***
	// This:
	// 1. Opens and parses GGUF file
	// 2. Allocates CPU/GPU memory for tensors
	// 3. Loads/mmaps weights into memory
	// 4. For Tesla K80: compiles CUDA kernels via PTX JIT (compute 3.7)
	slog.Info("LoadModelFromFile: calling llama_model_load_from_file (CGO -> C++)")
	cgoStart := time.Now()
	m := Model{c: C.llama_model_load_from_file(C.CString(modelPath), cparams)}
	slog.Info("LoadModelFromFile: llama_model_load_from_file returned", "duration_sec", time.Since(cgoStart).Seconds())

	if m.c == nil {
		return nil, fmt.Errorf("unable to load model: %s", modelPath)
	}

	slog.Info("LoadModelFromFile: COMPLETE", "total_duration_sec", time.Since(loadStart).Seconds())
	return &m, nil
}

func FreeModel(model *Model) {
	C.llama_model_free(model.c)
}

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	c := Context{
		c:          C.llama_init_from_model(model.c, params.c),
		numThreads: int(params.c.n_threads),
	}
	if c.c == nil {
		return nil, errors.New("unable to create llama context")
	}

	return &c, nil
}

func (m *Model) NumVocab() int {
	return int(C.llama_vocab_n_tokens(m.Vocab()))
}

func (m *Model) TokenIsEog(token int) bool {
	return bool(C.llama_vocab_is_eog(m.Vocab(), C.llama_token(token)))
}

func (m *Model) AddBOSToken() bool {
	return bool(C.llama_vocab_get_add_bos(m.Vocab()))
}

func (m *Model) ApplyLoraFromFile(context *Context, loraPath string, scale float32, threads int) error {
	cLoraPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cLoraPath))

	loraAdapter := C.llama_adapter_lora_init(m.c, cLoraPath)
	if loraAdapter == nil {
		return errors.New("unable to load lora")
	}

	err := -1
	if loraAdapter != nil {
		err = int(C.llama_set_adapter_lora(context.c, loraAdapter, C.float(scale)))
	}
	if err != 0 {
		return errors.New("error applying lora from file")
	}

	return nil
}

func (m *Model) Vocab() *C.struct_llama_vocab {
	return C.llama_model_get_vocab(m.c)
}

type Batch struct {
	c         C.struct_llama_batch
	batchSize int
	maxSeq    int
	embedSize int
}

// Creates a new batch for either word tokens or image embeddings (if embedSize is non-zero).
// Batches cannot contain both types at the same time. batchSize is the maximum number of entries
// that can be added per sequence
func NewBatch(batchSize int, maxSeq int, embedSize int) (*Batch, error) {
	b := Batch{
		c:         C.llama_batch_init(C.int(batchSize*maxSeq), C.int(embedSize), C.int(maxSeq)),
		batchSize: batchSize,
		maxSeq:    maxSeq,
		embedSize: embedSize,
	}

	// Check to see if any of the allocations in llama_batch_init() failed
	nilPointer := (embedSize == 0 && b.c.token == nil) || (embedSize != 0 && b.c.embd == nil) ||
		b.c.pos == nil || b.c.n_seq_id == nil || b.c.seq_id == nil || b.c.logits == nil ||
		slices.Contains(unsafe.Slice(b.c.seq_id, b.allocSize()), nil)

	if nilPointer {
		C.llama_batch_free(b.c)
		return nil, fmt.Errorf("unable to allocate batch (batchSize=%v maxSeq=%v embedSize=%v)", batchSize, maxSeq, embedSize)
	}

	return &b, nil
}

func (b *Batch) Size() int {
	return b.batchSize
}

func (b *Batch) allocSize() int {
	return b.batchSize * b.maxSeq
}

func (b *Batch) NumTokens() int {
	return int(b.c.n_tokens)
}

func (b *Batch) IsEmbedding() bool {
	return b.embedSize != 0
}

// Add adds either a token or an image embedding to the batch depending on the type
// when the batch was initialized. The other argument will be ignored. Adds to the
// batch with the given position for the given sequence ids, and optionally instructs
// to include logits.
func (b *Batch) Add(token int, embed []float32, pos int, logits bool, seqIds ...int) {
	if !b.IsEmbedding() {
		unsafe.Slice(b.c.token, b.allocSize())[b.c.n_tokens] = C.llama_token(token)
	} else {
		copy(unsafe.Slice((*float32)(b.c.embd), b.allocSize()*b.embedSize)[int(b.c.n_tokens)*b.embedSize:], embed)
	}
	unsafe.Slice(b.c.pos, b.allocSize())[b.c.n_tokens] = C.llama_pos(pos)
	unsafe.Slice(b.c.n_seq_id, b.allocSize())[b.c.n_tokens] = C.int(len(seqIds))

	for i, s := range seqIds {
		unsafe.Slice((unsafe.Slice(b.c.seq_id, b.allocSize())[b.c.n_tokens]), C.int(len(seqIds)))[i] = C.int32_t(s)
	}

	if logits {
		unsafe.Slice(b.c.logits, b.allocSize())[b.c.n_tokens] = 1
	} else {
		unsafe.Slice(b.c.logits, b.allocSize())[b.c.n_tokens] = 0
	}

	b.c.n_tokens += 1
}

func (b *Batch) Clear() {
	b.c.n_tokens = 0
}

func (b *Batch) Free() {
	b.batchSize = 0
	C.llama_batch_free(b.c)
}

type Model struct {
	c *C.struct_llama_model
}

func (m *Model) TokenToPiece(token int) string {
	tokenLen := 12
	buf := make([]byte, tokenLen)
	tokenLen = int(C.llama_token_to_piece(
		m.Vocab(),
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(tokenLen),
		C.int32_t(0),
		C.bool(true),
	))
	if tokenLen < 0 {
		tokenLen = -tokenLen

		buf = make([]byte, tokenLen)
		C.llama_token_to_piece(
			m.Vocab(),
			C.int32_t(token),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int32_t(tokenLen),
			C.int32_t(0),
			C.bool(true),
		)
	}
	return strings.TrimRight(string(buf), "\x00")
}

func (m *Model) Tokenize(text string, addSpecial bool, parseSpecial bool) ([]int, error) {
	maxTokens := len(text) + 2
	cTokens := make([]C.llama_token, maxTokens)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize(
		m.Vocab(),
		cText,
		C.int32_t(len(text)),
		&cTokens[0],
		C.int32_t(maxTokens),
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	// if the result is negative, reallocate and retry with the correct buffer size
	if result < 0 {
		maxTokens = int(-result)
		cTokens = make([]C.llama_token, maxTokens)
		result = C.llama_tokenize(
			m.Vocab(),
			cText,
			C.int32_t(len(text)),
			&cTokens[0],
			C.int32_t(maxTokens),
			C.bool(addSpecial),
			C.bool(parseSpecial),
		)
		if result < 0 {
			return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
		}
	}

	tokens := make([]int, result)
	for i := range result {
		tokens[i] = int(cTokens[i])
	}

	return tokens, nil
}

func (m *Model) NEmbd() int {
	return int(C.llama_model_n_embd(m.c))
}

// MemoryMeasurement holds actual measured memory requirements per backend device.
//
// MOTIVATION: The GraphSize() estimation formulas in fs/ggml/ggml.go often
// underestimate compute buffer requirements by 3-4×, causing allocation failures
// and wasted GPU capacity. This struct enables measurement-based GPU selection.
//
// ARCHITECTURE: Returned by MeasureMemoryRequirements() which creates a temporary
// llama.cpp context, allocates compute buffers, and queries actual sizes.
//
// CURRENT STATUS: API implemented but not yet integrated into GPU selection.
// Current solution uses 3.5× safety margin on estimates (llm/memory.go:377).
// This provides the foundation for future measurement-based approach.
type MemoryMeasurement struct {
	BackendName  string // Backend device name (e.g., "CUDA0", "CUDA1", "CPU")
	ModelBytes   uint64 // Model weights memory in bytes
	ContextBytes uint64 // KV cache memory in bytes
	ComputeBytes uint64 // Compute buffer memory (temp tensors) in bytes
	TotalBytes   uint64 // Total memory requirement in bytes
	IsHost       bool   // True if this is a host (CPU) backend
}

// MeasureMemoryRequirements measures actual memory requirements for this model
// with given context parameters. This allows the Go layer to make informed GPU
// selection decisions before committing to a configuration.
//
// This function creates a temporary context, reserves compute buffers, and queries
// actual memory requirements per backend. It handles allocation failures gracefully
// by retrieving attempted sizes even when allocation fails.
//
// Parameters:
//   - params: Context parameters (n_ctx, n_batch, n_ubatch, n_seq_max)
//
// Returns:
//   - []MemoryMeasurement: Per-backend memory breakdown
//   - error: Non-nil if measurement fails
//
// Example usage:
//
//	measurements, err := model.MeasureMemoryRequirements(params)
//	if err != nil {
//	    // Fallback to estimation
//	}
//	for _, m := range measurements {
//	    fmt.Printf("%s: %d MB total\n", m.BackendName, m.TotalBytes/1024/1024)
//	}
func (m *Model) MeasureMemoryRequirements(params ContextParams) ([]MemoryMeasurement, error) {
	const maxBackends = 16 // llama_max_devices() returns 16
	cMeasurements := make([]C.struct_llama_memory_measurement, maxBackends)

	numBackends := C.llama_measure_memory_requirements(
		m.c,
		params.c,
		&cMeasurements[0],
		C.int32_t(maxBackends),
	)

	if numBackends < 0 {
		return nil, fmt.Errorf("failed to measure memory requirements")
	}

	measurements := make([]MemoryMeasurement, numBackends)
	for i := range numBackends {
		measurements[i] = MemoryMeasurement{
			BackendName:  C.GoString(&cMeasurements[i].backend_name[0]),
			ModelBytes:   uint64(cMeasurements[i].model_bytes),
			ContextBytes: uint64(cMeasurements[i].context_bytes),
			ComputeBytes: uint64(cMeasurements[i].compute_bytes),
			TotalBytes:   uint64(cMeasurements[i].total_bytes),
			IsHost:       bool(cMeasurements[i].is_host),
		}
	}

	return measurements, nil
}

// vision processing
type MtmdContext struct {
	c *C.struct_mtmd_context
}

func NewMtmdContext(llamaContext *Context, modelPath string) (*MtmdContext, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))
	// TODO: Support non-default params
	cp := C.mtmd_context_params_default()

	// NOTE: The model and projector embedding lengths are checked during init
	c := C.mtmd_init_from_file(mp, C.llama_get_model(llamaContext.c), cp)
	if c == nil {
		return nil, fmt.Errorf("unable to load mmtd model: %v", modelPath)
	}

	return &MtmdContext{c: c}, nil
}

func (c *MtmdContext) Free() {
	C.mtmd_free(c.c)
}

type MtmdChunk struct {
	Embed  []float32
	Tokens []int
}

func (c *MtmdContext) MultimodalTokenize(llamaContext *Context, data []byte) ([]MtmdChunk, error) {
	// Initialize the input chunks pointer
	ic := C.mtmd_input_chunks_init()
	defer C.mtmd_input_chunks_free(ic)

	// Initialize an empty text prompt so we can tokenize
	it := C.mtmd_input_text_init(C.mtmd_default_marker(), true, true)
	defer C.mtmd_input_text_free(it)

	// Initialize a bitmap with the image data
	bm := C.mtmd_helper_bitmap_init_from_buf(c.c, (*C.uchar)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	defer C.mtmd_bitmap_free(bm)

	// Tokenize the image
	if C.int32_t(0) != C.mtmd_tokenize(c.c, ic, it, &bm, 1) {
		return nil, errors.New("unable to tokenize mtmd embedding from image")
	}
	nChunks := C.mtmd_input_chunks_size(ic)
	numEmbed := llamaContext.Model().NEmbd()
	outChunks := make([]MtmdChunk, 0)
	for i := range int(nChunks) {
		chunk := C.mtmd_input_chunks_get(ic, C.size_t(i))
		numTokens := int(C.mtmd_input_chunk_get_n_tokens(chunk))
		slog.Debug("chunk tokens", "index", i, "numTokens", numTokens)

		if C.mtmd_input_chunk_get_type(chunk) == C.MTMD_INPUT_CHUNK_TYPE_TEXT {
			// If this is a text chunk, add the tokens
			cNumTokens := C.size_t(0)
			cTokens := C.mtmd_input_chunk_get_tokens_text(chunk, &cNumTokens)
			cTokensArr := unsafe.Slice(cTokens, int(cNumTokens))
			tokens := make([]int, int(cNumTokens))
			for j := range int(cNumTokens) {
				tokens[j] = int(cTokensArr[j])
			}
			outChunks = append(outChunks, MtmdChunk{Tokens: tokens})
		} else {
			// Otherwise, encode the image chunk to embeddings

			// Encode the chunk
			if C.int32_t(0) != C.mtmd_encode_chunk(c.c, chunk) {
				return nil, errors.New("unable to encode mtmd image chunk")
			}

			// Get the embeddings for this chunk
			chunkEmbed := make([][]float32, numTokens)
			chunkEmbd := C.mtmd_get_output_embd(c.c)
			if nil == chunkEmbd {
				return nil, errors.New("no mtmd image embedding")
			}

			// Extend the embedding array for each token
			s := unsafe.Slice((*float32)(chunkEmbd), numTokens*numEmbed)
			rows := make([]float32, len(s))
			copy(rows, s)
			for i := range numTokens {
				chunkEmbed[i] = rows[i*numEmbed : (i+1)*numEmbed]
			}
			for _, e := range chunkEmbed {
				outChunks = append(outChunks, MtmdChunk{Embed: e})
			}
		}
	}
	slog.Debug("image tokenization chunks", "totalChunks", len(outChunks))
	return outChunks, nil
}

func (c *Context) Synchronize() {
	C.llama_synchronize(c.c)
}

// sampling
// TODO: this is a temporary wrapper to allow calling C++ code from CGo
type SamplingContext struct {
	c *C.struct_common_sampler
}

type SamplingParams struct {
	TopK           int
	TopP           float32
	MinP           float32
	TypicalP       float32
	Temp           float32
	RepeatLastN    int
	PenaltyRepeat  float32
	PenaltyFreq    float32
	PenaltyPresent float32
	PenalizeNl     bool
	Seed           uint32
	Grammar        string
}

func NewSamplingContext(model *Model, params SamplingParams) (*SamplingContext, error) {
	var cparams C.struct_common_sampler_cparams
	cparams.top_k = C.int32_t(params.TopK)
	cparams.top_p = C.float(params.TopP)
	cparams.min_p = C.float(params.MinP)
	cparams.typical_p = C.float(params.TypicalP)
	cparams.temp = C.float(params.Temp)
	cparams.penalty_last_n = C.int32_t(params.RepeatLastN)
	cparams.penalty_repeat = C.float(params.PenaltyRepeat)
	cparams.penalty_freq = C.float(params.PenaltyFreq)
	cparams.penalty_present = C.float(params.PenaltyPresent)
	cparams.seed = C.uint32_t(params.Seed)

	grammar := C.CString(params.Grammar)
	defer C.free(unsafe.Pointer(grammar))

	cparams.grammar = grammar
	context := &SamplingContext{c: C.common_sampler_cinit(model.c, &cparams)}
	if context.c == nil {
		return nil, errors.New("unable to create sampling context")
	}

	runtime.SetFinalizer(context, func(s *SamplingContext) { C.common_sampler_cfree(s.c) })

	return context, nil
}

func (s *SamplingContext) Reset() {
	C.common_sampler_creset(s.c)
}

func (s *SamplingContext) Sample(llamaContext *Context, idx int) int {
	return int(C.common_sampler_csample(s.c, llamaContext.c, C.int(idx)))
}

func (s *SamplingContext) Accept(id int, applyGrammar bool) {
	C.common_sampler_caccept(s.c, C.llama_token(id), C.bool(applyGrammar))
}

// SchemaToGrammar converts the provided JSON schema to a grammar. It returns
// nil if the provided schema is invalid JSON or an invalid JSON schema.
func SchemaToGrammar(schema []byte) []byte {
	cStr := C.CString(string(schema))
	defer C.free(unsafe.Pointer(cStr))

	// Allocate buffer for grammar based on schema length but with upper bound
	maxLen := max(32768, min(1024*1024, len(schema)*4))
	buf := make([]byte, maxLen)

	// Call C function to convert schema to grammar
	n := C.schema_to_grammar(cStr, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(maxLen))
	if n == 0 {
		// preserve nil
		return nil
	}
	return buf[:n]
}

type TokenData struct {
	ID    int32
	Logit float32
}

type Grammar struct {
	c  *C.struct_llama_grammar
	mu sync.Mutex
}

func NewGrammar(grammar string, vocabIds []uint32, vocabValues []string, eogTokens []int32) *Grammar {
	cGrammar := C.CString(grammar)
	defer C.free(unsafe.Pointer(cGrammar))

	cTokens := make([]C.uint32_t, len(vocabIds))
	for i, token := range vocabIds {
		cTokens[i] = C.uint32_t(token)
	}

	cPieces := make([]*C.char, len(vocabValues))
	for i, piece := range vocabValues {
		cPieces[i] = C.CString(piece)
		defer C.free(unsafe.Pointer(cPieces[i]))
	}

	cEogTokens := make([]C.uint32_t, len(eogTokens))
	for i, token := range eogTokens {
		cEogTokens[i] = C.uint32_t(token)
	}

	g := C.grammar_init(cGrammar, unsafe.SliceData(cTokens), C.size_t(len(cTokens)), unsafe.SliceData(cPieces), unsafe.SliceData(cEogTokens), C.size_t(len(cEogTokens)))
	if g == nil {
		return nil
	}

	return &Grammar{c: g}
}

func (g *Grammar) Free() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.c != nil {
		C.grammar_free(g.c)
		g.c = nil
	}
}

func (g *Grammar) Apply(tokens []TokenData) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.c == nil {
		return
	}

	tds := make([]C.struct_llama_token_data, len(tokens))
	for i, token := range tokens {
		tds[i] = C.struct_llama_token_data{
			id:    C.int32_t(token.ID),
			logit: C.float(token.Logit),
			p:     C.float(0.0),
		}
	}
	tda := &C.llama_token_data_array{
		data:     (*C.struct_llama_token_data)(unsafe.Pointer(&tds[0])),
		size:     C.size_t(len(tokens)),
		selected: C.int64_t(-1),
		sorted:   C.bool(false),
	}
	var pinner runtime.Pinner
	pinner.Pin(&tds[0])
	defer pinner.Unpin()

	C.grammar_apply(g.c, tda)
	for i := range tokens {
		tokens[i].Logit = float32(tds[i].logit)
	}
}

func (g *Grammar) Accept(token int32) {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Check if grammar was freed
	if g.c == nil {
		return
	}

	C.grammar_accept(g.c, C.llama_token(token))
}
