package llm

import (
	"fmt"
	"log/slog"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// modelFamilyBatchDefaults provides model-architecture-specific batch size hints.
// These are optimal batch sizes based on model architecture characteristics.
// Used when GGUF metadata doesn't specify batch sizes and user hasn't overridden.
//
// Key factors per architecture:
// - Attention mechanism (MHA, MQA, GQA) affects compute buffer size
// - FFN size and architecture affects memory patterns
// - Typical use cases (chat, completion, embedding)
//
// Architecture names match GGML's "general.architecture" field in GGUF.
type modelFamilyBatchParams struct {
	nBatch  uint32 // Logical batch size (max tokens to process at once)
	nUbatch uint32 // Physical batch size (micro-batch for memory efficiency)
}

var modelFamilyBatchDefaults = map[string]modelFamilyBatchParams{
	// DeepSeek models use large batches due to efficient MLA (Multi-head Latent Attention)
	"deepseek2": {nBatch: 2048, nUbatch: 256},

	// Llama family (standard transformer architecture)
	"llama":  {nBatch: 512, nUbatch: 512},  // Llama 2, Llama 3
	"llama4": {nBatch: 512, nUbatch: 512},  // Llama 4
	"mllama": {nBatch: 512, nUbatch: 512},  // Llama with vision encoder

	// Gemma family (efficient attention, similar to Llama)
	"gemma":   {nBatch: 512, nUbatch: 512},
	"gemma2":  {nBatch: 512, nUbatch: 512},
	"gemma3":  {nBatch: 512, nUbatch: 512},
	"gemma3n": {nBatch: 512, nUbatch: 512},

	// Qwen family (optimized for long context)
	"qwen2":    {nBatch: 512, nUbatch: 512},
	"qwen25vl": {nBatch: 512, nUbatch: 512}, // Qwen vision-language

	// Mistral family (sliding window attention)
	"mistral3": {nBatch: 512, nUbatch: 512},

	// Command-R (Cohere's architecture)
	"command-r": {nBatch: 512, nUbatch: 512},

	// Phi family (Microsoft's small models)
	"phi2": {nBatch: 256, nUbatch: 256}, // Smaller model, smaller batches

	// StableLM
	"stablelm": {nBatch: 512, nUbatch: 512},

	// ChatGLM (GLM architecture)
	"chatglm": {nBatch: 512, nUbatch: 512},

	// GPT-OSS (open-source GPT implementations)
	"gptoss":  {nBatch: 512, nUbatch: 512},
	"gpt-oss": {nBatch: 512, nUbatch: 512},
}

// getModelBatchParams returns optimal batch parameters for a model.
// Priority order:
// 1. User-specified values (via api.Options)
// 2. Model family defaults (based on architecture)
// 3. Global defaults (512/512)
func getModelBatchParams(architecture string, opts api.Options) (nBatch, nUbatch uint32) {
	// Use user-specified batch size if provided
	nBatch = uint32(opts.NumBatch)
	if nBatch == 0 {
		// Try model family default
		if params, ok := modelFamilyBatchDefaults[architecture]; ok {
			nBatch = params.nBatch
			nUbatch = params.nUbatch
			slog.Debug("using model family batch defaults",
				"architecture", architecture,
				"n_batch", nBatch,
				"n_ubatch", nUbatch)
			return nBatch, nUbatch
		}
		// Global default
		nBatch = 512
	}

	// nUbatch defaults to nBatch if not in family defaults
	nUbatch = nBatch

	return nBatch, nUbatch
}

// pickBestFullFitByLibrary will try to find the optimal placement of the model in the available GPUs where the model fully fits
// The list of GPUs returned will always be the same brand (library)
// If the model can not be fit fully within the available GPU(s) nil is returned
func pickBestFullFitByLibrary(f *ggml.GGML, modelPath string, projectors []string, adapters []string, opts api.Options, gpus []ml.DeviceInfo, numParallel int) []ml.DeviceInfo {
	for _, gl := range ml.ByLibrary(gpus) {
		sgl := append(make([]ml.DeviceInfo, 0, len(gl)), gl...)

		// TODO - potentially sort by performance capability, existing models loaded, etc.
		// TODO - Eliminate any GPUs that already have envconfig.MaxRunners loaded on them
		// Note: at present, this will favor most current available VRAM descending and ignoring faster GPU speed in mixed setups
		sort.Sort(sort.Reverse(ml.ByFreeMemory(sgl)))

		if !envconfig.SchedSpread() {
			// Try to pack into as few GPUs as possible, starting from 1 GPU
			for numGPUs := 1; numGPUs <= len(sgl); numGPUs++ {
				gpuSubset := sgl[:numGPUs]
				ok, estimatedVRAM := predictServerFit(gpuSubset, f, adapters, projectors, opts, numParallel)

				if ok {
					slog.Info("new model will fit in available VRAM across minimum required GPUs, loading",
						"model", modelPath,
						"library", sgl[0].Library,
						"parallel", numParallel,
						"required", format.HumanBytes2(estimatedVRAM),
						"gpus", numGPUs)
					return gpuSubset
				}
			}
		} else {
			// TODO future refinements
			// - if multiple Libraries, see if any single GPU in any Library will fit
			// - try subsets of GPUs instead of just falling back to 1 or all in a family

			// Now try all the GPUS (OLLAMA_SCHED_SPREAD is set)
			if ok, estimatedVRAM := predictServerFit(sgl, f, adapters, projectors, opts, numParallel); ok {
				slog.Info("new model will fit in available VRAM, loading",
					"model", modelPath,
					"library", sgl[0].Library,
					"parallel", numParallel,
					"required", format.HumanBytes2(estimatedVRAM),
					"gpus", len(sgl))
				return sgl
			}
		}
	}
	return nil
}

// If multiple Libraries are detected, pick the Library which loads the most layers for the model
func pickBestPartialFitByLibrary(f *ggml.GGML, projectors []string, adapters []string, opts api.Options, gpus []ml.DeviceInfo, numParallel int) []ml.DeviceInfo {
	byLibrary := ml.ByLibrary(gpus)
	if len(byLibrary) <= 1 {
		return gpus
	}
	var bestEstimate uint64
	var bestFit int
	for i, gl := range byLibrary {
		_, estimatedVRAM := predictServerFit(gl, f, adapters, projectors, opts, numParallel)
		if estimatedVRAM > bestEstimate {
			bestEstimate = estimatedVRAM
			bestFit = i
		}
	}
	return byLibrary[bestFit]
}

// This algorithm looks for a complete fit to determine if we need to unload other models
func predictServerFit(allGpus []ml.DeviceInfo, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (bool, uint64) {
	// Split up the GPUs by type and try them
	var estimatedVRAM uint64
	for _, gpus := range ml.ByLibrary(allGpus) {
		var layerCount int
		estimate := estimateGPULayers(gpus, f, projectors, opts, numParallel)
		layerCount, estimatedVRAM = estimate.Layers, estimate.VRAMSize
		if opts.NumGPU < 0 {
			if layerCount > 0 && layerCount >= int(f.KV().BlockCount()+1) {
				return true, estimatedVRAM
			}
		} else {
			if layerCount > 0 && layerCount >= opts.NumGPU {
				return true, estimatedVRAM
			}
		}
	}
	return false, estimatedVRAM
}

func verifyCPUFit(f *ggml.GGML, modelPath string, projectors []string, adapters []string, opts api.Options, systemInfo ml.SystemInfo, numParallel int) bool {
	estimate := estimateGPULayers(nil, f, projectors, opts, numParallel)
	if estimate.TotalSize > systemInfo.FreeMemory {
		return false
	}
	slog.Info("new model will fit in available system memory for CPU inference, loading",
		"model", modelPath,
		"parallel", numParallel,
		"required", format.HumanBytes2(estimate.TotalSize),
	)
	return true
}

type MemoryEstimate struct {
	// How many layers we predict we can load
	Layers int

	// The size of the graph which occupies the main GPU
	Graph uint64

	// How much VRAM will be allocated given the number of layers we predict
	VRAMSize uint64

	// The total size of the model if loaded into VRAM.  If all layers are loaded, VRAMSize == TotalSize
	TotalSize uint64

	// For multi-GPU scenarios, this provides the tensor split parameter
	TensorSplit []int

	// For multi-GPU scenarios, this is the size in bytes per GPU
	GPUSizes []uint64

	// internal fields for logging purposes
	inferenceLibrary    string
	layersRequested     int
	layersModel         int
	availableList       []string
	kv                  uint64
	allocationsList     []string
	memoryWeights       uint64
	memoryLayerOutput   uint64
	graphFullOffload    uint64
	graphPartialOffload uint64

	projectorWeights, projectorGraph uint64
}

// Given a model and one or more GPU targets, predict how many layers and bytes we can load, and the total size
// The GPUs provided must all be the same Library
func estimateGPULayers(gpus []ml.DeviceInfo, f *ggml.GGML, projectors []string, opts api.Options, numParallel int) MemoryEstimate {
	// Graph size for a partial offload, applies to all GPUs
	var graphPartialOffload uint64

	// Graph size when all layers are offloaded, applies to all GPUs
	var graphFullOffload uint64

	// Final graph offload once we know full or partial
	var graphOffload uint64

	// Projectors loaded into GPU0 only
	var llamaEngineProjectorWeights uint64

	// Projectors loaded with output layer
	var ollamaEngineProjectorWeights uint64
	var ollamaEngineProjectorGraph uint64

	// Conditional output size on GPU 0
	var memoryLayerOutput uint64

	// The sizes of a layer
	var layerSize uint64

	// The sum of all the layer sizes (just for logging)
	var memoryWeights uint64

	// True if all the layers are loaded
	var fullyLoaded bool

	// Overflow that didn't fit into the GPU
	var overflow uint64

	overhead := envconfig.GpuOverhead()
	availableList := make([]string, len(gpus))
	libraries := []string{}
	for i, gpu := range gpus {
		availableList[i] = format.HumanBytes2(gpu.FreeMemory)
		if !slices.Contains(libraries, gpu.Library) {
			libraries = append(libraries, gpu.Library)
		}
	}
	if len(libraries) == 0 {
		libraries = []string{"cpu"}
	}
	slog.Debug("evaluating", "library", strings.Join(libraries, ","), "gpu_count", len(gpus), "available", availableList)

	for _, projector := range projectors {
		llamaEngineProjectorWeights += projectorMemoryRequirements(projector)
	}
	if llamaEngineProjectorWeights == 0 {
		ollamaEngineProjectorWeights, ollamaEngineProjectorGraph = f.VisionGraphSize()
	}

	layers := f.Tensors().GroupLayers()
	// add one layer worth of memory as a buffer
	if blk0, ok := layers["blk.0"]; ok {
		layerSize = blk0.Size()
	} else {
		slog.Warn("model missing blk.0 layer size")
	}

	useFlashAttention := envconfig.FlashAttention(f.FlashAttention()) &&
		ml.FlashAttentionSupported(gpus) &&
		f.SupportsFlashAttention()

	var kvct string
	if useFlashAttention {
		requested := strings.ToLower(envconfig.KvCacheType())
		if f.SupportsKVCacheType(requested) {
			kvct = requested
		}
	}

	// Get architecture-appropriate batch size for accurate memory estimation.
	//
	// WHY THIS MATTERS: GraphSize() compute buffer formulas scale linearly with batch size.
	// Using wrong batch size causes estimation errors that compound with model size.
	//
	// Example calculation (from fs/ggml/ggml.go qwen2 formula line 717):
	//   compute_buffer = 4 * batch * (2 + 3*embedding + context*(1+heads))
	//
	// For deepseek-r1:14b (qwen2 architecture):
	//   - Wrong batch (512): 4 * 512 * (...) ≈ 916 MB
	//   - Correct batch (2048): 4 * 2048 * (...) ≈ 3.7 GB
	//   - Difference: 4× underestimation!
	//
	// Different architectures have different optimal batch sizes based on:
	// - Attention mechanism efficiency (MHA vs GQA vs MLA)
	// - FFN architecture (standard vs gated vs MoE)
	// - Typical inference patterns (chat vs completion vs embedding)
	//
	// Priority: User override > Model family default > Global default (512)
	architecture := f.KV().Architecture()
	nBatch, _ := getModelBatchParams(architecture, opts)

	// Cap batch size at context length (can't process more tokens than context)
	batchSize := min(uint64(opts.NumCtx), uint64(nBatch))

	slog.Debug("estimating memory with model-specific batch size",
		"architecture", architecture,
		"n_batch", nBatch,
		"effective_batch", batchSize,
		"n_ctx", opts.NumCtx)

	kv, graphPartialOffload, graphFullOffload := f.GraphSize(uint64(opts.NumCtx), batchSize, numParallel, kvct, useFlashAttention)

	if len(kv) > 0 {
		layerSize += kv[0]
	}

	var kvTotal uint64
	for _, kvLayer := range kv {
		kvTotal += kvLayer
	}

	if graphPartialOffload == 0 {
		headsKV := f.KV().HeadCountKVMin()
		if headsKV == 0 {
			headsKV = 1
		}
		gqa := f.KV().HeadCountMax() / headsKV
		graphPartialOffload = gqa * kvTotal / 6
	}
	if graphFullOffload == 0 {
		graphFullOffload = graphPartialOffload
	}

	// Apply safety margin for compute buffers to account for formula inaccuracies.
	//
	// PROBLEM: The GraphSize() formulas in fs/ggml/ggml.go are mathematical estimates
	// that don't account for all temporary tensors allocated during inference.
	// These formulas were derived from the architecture specifications, but actual
	// llama.cpp inference allocates additional intermediate buffers for:
	// - Attention score matrices (Q*K^T)
	// - Intermediate FFN activations
	// - Gradient accumulation buffers
	// - Temporary workspace for CUDA operations
	//
	// ROOT CAUSE ANALYSIS (deepseek-r1:14b case study):
	// - Model: 14B parameters (qwen2 architecture)
	// - Estimated compute buffer: 916 MB (from GraphSize formula)
	// - Actual allocation attempt: ~3-4 GB (observed from allocation failure)
	// - Underestimation factor: 3.3-4.4×
	//
	// The underestimation gets worse for:
	// - Larger models (>10B parameters): More layers = more intermediate tensors
	// - Larger batch sizes (>512): Batch dimension multiplies intermediate tensor sizes
	// - Grouped-query attention (GQA): Complex attention patterns need more workspace
	// - MoE architectures: Multiple expert activations need simultaneous storage
	//
	// SOLUTION: Apply 3.5× conservative safety margin to prevent allocation failures.
	// This ensures GPU selection uses realistic memory requirements, enabling proper
	// multi-GPU distribution when needed.
	//
	// TRADE-OFF: May cause some models to use 2 GPUs when 1 GPU might suffice,
	// but prevents catastrophic allocation failures. Future improvement: implement
	// measurement-based approach using llama_measure_memory_requirements() API.
	graphSafetyMultiplier := 3.5
	graphPartialOffload = uint64(float64(graphPartialOffload) * graphSafetyMultiplier)
	graphFullOffload = uint64(float64(graphFullOffload) * graphSafetyMultiplier)

	slog.Debug("applied compute buffer safety margin",
		"multiplier", graphSafetyMultiplier,
		"graph_partial_offload", format.HumanBytes2(graphPartialOffload),
		"graph_full_offload", format.HumanBytes2(graphFullOffload))

	// on metal there's no partial offload overhead
	if len(gpus) > 0 && gpus[0].Library == "Metal" {
		graphPartialOffload = graphFullOffload
	} else if len(gpus) > 1 {
		// multigpu should always use the partial graph size
		graphFullOffload = graphPartialOffload
	}

	// Output layer handled at the end if we have space
	if layer, ok := layers["output_norm"]; ok {
		memoryLayerOutput += layer.Size()
	}
	if layer, ok := layers["output"]; ok {
		memoryLayerOutput += layer.Size()
	} else if layer, ok := layers["token_embd"]; ok {
		memoryLayerOutput += layer.Size()
	}

	gpuZeroOverhead := llamaEngineProjectorWeights

	// Reduce set of GPUs to only those that have sufficient space to fit overhead and at least one layer
	var layerCount int
	tensorSplit := make([]int, len(gpus))
	gpuAllocations := make([]uint64, len(gpus))
	type gs struct {
		i int
		g *ml.DeviceInfo
	}
	gpusWithSpace := []gs{}
	for i := range gpus {
		var gzo uint64
		if len(gpusWithSpace) == 0 {
			gzo = gpuZeroOverhead
		}
		// Only include GPUs that can fit the graph, gpu minimum, the layer buffer and at least more layer
		if gpus[i].FreeMemory < overhead+gzo+max(graphPartialOffload, graphFullOffload)+gpus[i].MinimumMemory()+2*layerSize {
			slog.Debug("gpu has too little memory to allocate any layers",
				"id", gpus[i].ID,
				"library", gpus[i].Library,
				"compute", gpus[i].Compute(),
				"driver", fmt.Sprintf("%d.%d", gpus[i].DriverMajor, gpus[i].DriverMinor),
				"name", gpus[i].Name,
				"total", format.HumanBytes2(gpus[i].TotalMemory),
				"available", format.HumanBytes2(gpus[i].FreeMemory),
				"minimum_memory", gpus[i].MinimumMemory,
				"layer_size", format.HumanBytes2(layerSize),
				"gpu_zer_overhead", format.HumanBytes2(gzo),
				"partial_offload", format.HumanBytes2(graphPartialOffload),
				"full_offload", format.HumanBytes2(graphFullOffload),
			)
			continue
		}
		gpusWithSpace = append(gpusWithSpace, gs{i, &gpus[i]})
		gpuAllocations[i] += gpus[i].MinimumMemory() + layerSize // We hold off on graph until we know partial vs. full
	}

	var gpuZeroID int
	if len(gpusWithSpace) > 0 {
		gpuZeroID = gpusWithSpace[0].i
		gpuAllocations[gpuZeroID] += gpuZeroOverhead
	} else {
		overflow += gpuZeroOverhead
	}

	// For all the layers, find where they can fit on the GPU(s)
	for i := int(f.KV().BlockCount()) - 1; i >= 0; i-- {
		// Some models have inconsistent layer sizes
		if blk, ok := layers[fmt.Sprintf("blk.%d", i)]; ok {
			layerSize = blk.Size()
			layerSize += kv[i]
			memoryWeights += blk.Size()
		}

		if opts.NumGPU >= 0 && layerCount >= opts.NumGPU {
			// Stop allocating on GPU(s) once we hit the users target NumGPU
			overflow += layerSize
			continue
		}

		// distribute the layers across the GPU(s) that have space
		for j := len(gpusWithSpace); j > 0; j-- {
			g := gpusWithSpace[i%j]
			used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
			if g.g.FreeMemory > overhead+used+layerSize {
				gpuAllocations[g.i] += layerSize
				tensorSplit[g.i]++
				layerCount++
				break
			} else {
				gpusWithSpace = append(gpusWithSpace[:i%j], gpusWithSpace[i%j+1:]...)
			}
		}

		if len(gpusWithSpace) == 0 {
			overflow += layerSize
		}
	}
	if layerCount >= int(f.KV().BlockCount()) {
		fullyLoaded = true
	}

	// Determine if we need to consider output then find where it fits
	memoryLastLayer := memoryLayerOutput + ollamaEngineProjectorWeights + ollamaEngineProjectorGraph
	if memoryLastLayer > 0 {
		if opts.NumGPU < 0 || layerCount < opts.NumGPU {
			for j := len(gpusWithSpace); j > 0; j-- {
				g := gpusWithSpace[layerCount%j]
				used := gpuAllocations[g.i] + max(graphPartialOffload, graphFullOffload)
				if g.g.FreeMemory > overhead+used+memoryLastLayer {
					gpuAllocations[g.i] += memoryLastLayer
					tensorSplit[g.i]++
					layerCount++
					break
				}
			}
		}

		if layerCount < int(f.KV().BlockCount())+1 {
			fullyLoaded = false
			overflow += memoryLastLayer
		}
	}

	// Add the applicable (full or partial) graph allocations
	for i := range gpus {
		if tensorSplit[i] <= 0 {
			continue
		}
		if fullyLoaded {
			gpuAllocations[i] += graphFullOffload
		} else {
			gpuAllocations[i] += graphPartialOffload
		}
	}
	if fullyLoaded {
		graphOffload = graphFullOffload
	} else {
		graphOffload = graphPartialOffload
	}

	// Summaries for the log
	var memoryRequiredPartial, memoryRequiredTotal uint64
	for i := range gpuAllocations {
		memoryRequiredPartial += gpuAllocations[i]
	}
	memoryRequiredTotal = memoryRequiredPartial + overflow

	allocationsList := []string{}
	for _, a := range gpuAllocations {
		allocationsList = append(allocationsList, format.HumanBytes2(a))
	}

	estimate := MemoryEstimate{
		TotalSize: memoryRequiredTotal,
		Layers:    0,
		Graph:     0,
		VRAMSize:  0,
		GPUSizes:  []uint64{},

		inferenceLibrary:    strings.Join(libraries, ","),
		layersRequested:     opts.NumGPU,
		layersModel:         int(f.KV().BlockCount()) + 1,
		availableList:       availableList,
		kv:                  kvTotal,
		allocationsList:     allocationsList,
		memoryWeights:       memoryWeights,
		memoryLayerOutput:   memoryLayerOutput,
		graphFullOffload:    graphFullOffload,
		graphPartialOffload: graphPartialOffload,
		projectorWeights:    llamaEngineProjectorWeights + ollamaEngineProjectorWeights,
		projectorGraph:      ollamaEngineProjectorGraph,
	}

	if len(gpus) == 0 {
		return estimate
	}
	if layerCount == 0 {
		slog.Debug("insufficient VRAM to load any model layers")
		return estimate
	}
	estimate.Layers = layerCount
	estimate.Graph = graphOffload
	estimate.VRAMSize = memoryRequiredPartial
	estimate.TotalSize = memoryRequiredTotal
	estimate.TensorSplit = tensorSplit
	estimate.GPUSizes = gpuAllocations
	return estimate
}

func (m MemoryEstimate) LogValue() slog.Value {
	attrs := []slog.Attr{
		slog.String("library", m.inferenceLibrary),
		slog.Group(
			"layers",
			// requested number of layers to offload
			"requested", m.layersRequested,
			// The number of layers the model has (including output)
			"model", m.layersModel,
			// estimated number of layers that can be offloaded
			"offload", m.Layers,
			// multi-gpu split for tensors
			"split", m.TensorSplit,
		),
		slog.Group(
			"memory",
			// memory available by GPU for offloading
			"available", m.availableList,
			"gpu_overhead", format.HumanBytes2(envconfig.GpuOverhead()),
			slog.Group(
				"required",
				// memory required for full offloading
				"full", format.HumanBytes2(m.TotalSize),
				// memory required to offload layers.estimate layers
				"partial", format.HumanBytes2(m.VRAMSize),
				// memory of KV cache
				"kv", format.HumanBytes2(m.kv),
				// Allocations across the GPUs
				"allocations", m.allocationsList,
			),
			slog.Group(
				"weights",
				// memory of the weights
				"total", format.HumanBytes2(m.memoryWeights+m.memoryLayerOutput),
				// memory of repeating layers
				"repeating", format.HumanBytes2(m.memoryWeights),
				// memory of non-repeating layers
				"nonrepeating", format.HumanBytes2(m.memoryLayerOutput),
			),
			slog.Group(
				"graph",
				// memory of graph when fully offloaded
				"full", format.HumanBytes2(m.graphFullOffload),
				// memory of graph when not fully offloaded
				"partial", format.HumanBytes2(m.graphPartialOffload),
			),
		),
	}

	if m.projectorWeights > 0 {
		attrs = append(attrs, slog.Group(
			"projector",
			"weights", format.HumanBytes2(m.projectorWeights),
			"graph", format.HumanBytes2(m.projectorGraph),
		))
	}

	return slog.GroupValue(attrs...)
}

func projectorMemoryRequirements(filename string) (weights uint64) {
	file, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer file.Close()

	ggml, err := ggml.Decode(file, 1024)
	if err != nil {
		return 0
	}

	for _, layer := range ggml.Tensors().GroupLayers() {
		weights += layer.Size()
	}

	return weights
}
