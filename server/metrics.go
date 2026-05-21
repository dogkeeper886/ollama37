package server

import (
	"context"
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/ml"
)

// serverMetrics holds atomic counters for server-wide events that are otherwise
// only visible via logs. Counters are monotonic; handlers expose them as-is.
// Per-model / per-request metrics are snapshotted on demand from the scheduler
// and do not live here.
type serverMetrics struct {
	loadFailures    atomic.Uint64 // NewLlamaServer returned an error
	loadRequireFull atomic.Uint64 // Load returned ErrLoadRequiredFull and was surfaced to the user (not counting scheduler retries)
	loadOther       atomic.Uint64 // Load returned a non-ErrLoadRequiredFull error
	evictionsTotal  atomic.Uint64 // runner unloaded — counts both idle-timer expirations and make-room evictions
}

// MetricsResponse is the JSON payload returned by GET /api/metrics.
type MetricsResponse struct {
	GPUs   []GPUMetrics   `json:"gpus"`
	Models []ModelMetrics `json:"models"`
	Errors ErrorCounters  `json:"errors"`
	Totals ServerTotals   `json:"totals"`
}

type GPUMetrics struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	VRAMTotal   uint64 `json:"vram_total"`
	VRAMFree    uint64 `json:"vram_free,omitempty"`
	ComputeCap  string `json:"compute_cap,omitempty"` // e.g. "3.7"
	LibraryPath string `json:"library_path,omitempty"`

	// Allocation attribution (derived). VRAMUsed is the physical usage on the
	// device (total - free, from NVML). VRAMAccounted is the sum that loaded
	// models report placing here. VRAMUnaccounted = used - accounted is
	// allocation the model layout does not explain — CUDA primary contexts,
	// "ghost" allocations, or other processes sharing the device.
	VRAMUsed        uint64 `json:"vram_used,omitempty"`
	VRAMAccounted   uint64 `json:"vram_accounted,omitempty"`
	VRAMUnaccounted uint64 `json:"vram_unaccounted,omitempty"`
	// HasModelLayers is true when a loaded model placed layers on this device.
	// The labeled components of the unaccounted memory (context, cuBLAS) are
	// reported per model in ModelMetrics.VRAMBreakdown (#98), since they are
	// only measurable in the model-runner process.
	HasModelLayers bool `json:"has_model_layers"`
	// Ghost flags a device carrying physical usage above idle while holding no
	// model layers — the signature of the 93 MiB ghost allocation (#98/#138)
	// that wakes an otherwise-idle GPU. NOTE: VRAMUsed is physical and includes
	// any other process on the device, so treat Ghost as a flag to investigate,
	// not proof of an ollama-created context.
	Ghost bool `json:"ghost,omitempty"`
}

type ModelMetrics struct {
	Name          string                    `json:"name"`
	Engine        string                    `json:"engine"`                   // "ollama" | "llamacpp"
	GPUs          []string                  `json:"gpus"`                     // device IDs
	VRAMTotal     uint64                    `json:"vram_total"`               // sum across GPUs
	VRAMByGPU     map[string]uint64         `json:"vram_by_gpu,omitempty"`    // per-device total
	VRAMBreakdown map[string]*VRAMBreakdown `json:"vram_breakdown,omitempty"` // ollama engine only, keyed by device ID
	ContextLength int                       `json:"context_length,omitempty"`
}

// VRAMBreakdown decomposes VRAM usage into its three components. Only the Go
// engine (ollamaServer) tracks this; for llamacpp models it is omitted.
type VRAMBreakdown struct {
	Weights uint64 `json:"weights"`           // model weights on GPU
	KVCache uint64 `json:"kv_cache"`          // K/V attention cache
	Graph   uint64 `json:"graph"`             // scratch compute buffer
	Context uint64 `json:"context,omitempty"` // CUDA primary context + driver/kernel baseline (#98)
	Cublas  uint64 `json:"cublas,omitempty"`  // cuBLAS GEMM workspace (#98)
}

type ErrorCounters struct {
	LoadFailures    uint64 `json:"load_failures"`     // runner process creation failed
	LoadRequireFull uint64 `json:"load_require_full"` // model didn't fit, surfaced to user (excludes scheduler retries)
	LoadOther       uint64 `json:"load_other"`        // any other load error
	EvictionsTotal  uint64 `json:"evictions_total"`   // runner unloaded (idle OR make-room; see design notes)
}

type ServerTotals struct {
	LoadedModels int `json:"loaded_models"`
	GPUCount     int `json:"gpu_count"`
}

func (s *Server) MetricsHandler(c *gin.Context) {
	s.sched.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.sched.loaded))
	for _, r := range s.sched.loaded {
		runners = append(runners, r)
	}
	s.sched.loadedMu.Unlock()

	// Discover GPUs via the same path the scheduler uses (discover.GPUDevices).
	// Going through the scheduler — instead of asking a loaded runner — means
	// the GPU list is populated even when no model is loaded, which is the
	// state most operators check first. Cap with a short timeout so a slow
	// discovery cannot hang the endpoint.
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	runnersForDiscovery := make([]ml.FilteredRunnerDiscovery, 0, len(runners))
	for _, r := range runners {
		runnersForDiscovery = append(runnersForDiscovery, r)
	}
	devices := s.sched.getGpuFn(ctx, runnersForDiscovery)

	gpus := make([]GPUMetrics, 0, len(devices))
	for _, d := range devices {
		compute := ""
		// ComputeMajor == -1 means the backend didn't report capability;
		// real GPUs are always >= 1. Skip the "0.0" edge case.
		if d.ComputeMajor > 0 {
			compute = gpuComputeString(d.ComputeMajor, d.ComputeMinor)
		}
		libPath := ""
		if len(d.LibraryPath) > 0 {
			libPath = d.LibraryPath[0]
		}
		// Prefer the user-friendly Description (e.g. "Tesla K80") over the
		// backend's Name field, which is the runtime enumeration label
		// (e.g. "CUDA0"). Name is what humans look at when answering "what
		// hardware is this?" — Description matches that intent. Fall back
		// to Name if Description is empty (some backends may not set it).
		name := d.Description
		if name == "" {
			name = d.Name
		}
		gpus = append(gpus, GPUMetrics{
			ID:          d.DeviceID.ID,
			Name:        name,
			VRAMTotal:   d.TotalMemory,
			VRAMFree:    d.FreeMemory,
			ComputeCap:  compute,
			LibraryPath: libPath,
		})
	}

	models := make([]ModelMetrics, 0, len(runners))
	for _, r := range runners {
		if r.llama == nil || r.model == nil {
			continue
		}

		gpuIDs := make([]string, 0, len(r.gpus))
		vramByGPU := make(map[string]uint64, len(r.gpus))
		for _, g := range r.gpus {
			gpuIDs = append(gpuIDs, g.ID)
			vramByGPU[g.ID] = r.llama.VRAMByGPU(g)
		}

		m := ModelMetrics{
			Name:      r.model.ShortName,
			Engine:    r.llama.Engine(),
			GPUs:      gpuIDs,
			VRAMTotal: r.vramSize,
			VRAMByGPU: vramByGPU,
		}
		if r.Options != nil {
			m.ContextLength = r.Options.NumCtx
		}
		if mem := r.llama.MemoryBreakdown(); mem != nil {
			m.VRAMBreakdown = perGPUBreakdown(mem)
		}
		models = append(models, m)
	}

	attributeGPUUsage(gpus, models)

	c.JSON(http.StatusOK, MetricsResponse{
		GPUs:   gpus,
		Models: models,
		Errors: s.sched.metrics.snapshot(),
		Totals: ServerTotals{
			LoadedModels: len(runners),
			GPUCount:     len(devices),
		},
	})
}

func (m *serverMetrics) snapshot() ErrorCounters {
	if m == nil {
		return ErrorCounters{}
	}
	return ErrorCounters{
		LoadFailures:    m.loadFailures.Load(),
		LoadRequireFull: m.loadRequireFull.Load(),
		LoadOther:       m.loadOther.Load(),
		EvictionsTotal:  m.evictionsTotal.Load(),
	}
}

// attributeGPUUsage fills the derived per-GPU attribution fields (VRAMUsed,
// VRAMAccounted, VRAMUnaccounted, HasModelLayers, Ghost) in place, from physical
// device usage and what loaded models report placing on each device. A device
// carrying usage above idle while holding no model layers is flagged as a ghost
// to investigate (#98/#138). VRAMUsed is physical, so it includes any other
// process on the device — Ghost is a flag, not proof of an ollama context.
func attributeGPUUsage(gpus []GPUMetrics, models []ModelMetrics) {
	const idleBaselineBytes = 20 * 1024 * 1024 // K80 idle is ~4 MiB; allow margin
	accountedByGPU := make(map[string]uint64, len(gpus))
	layersByGPU := make(map[string]bool, len(gpus))
	for _, m := range models {
		for id, v := range m.VRAMByGPU {
			accountedByGPU[id] += v
		}
		for _, id := range m.GPUs {
			layersByGPU[id] = true
		}
	}
	for i := range gpus {
		g := &gpus[i]
		if g.VRAMTotal > g.VRAMFree {
			g.VRAMUsed = g.VRAMTotal - g.VRAMFree
		}
		g.VRAMAccounted = accountedByGPU[g.ID]
		if g.VRAMUsed > g.VRAMAccounted {
			g.VRAMUnaccounted = g.VRAMUsed - g.VRAMAccounted
		}
		g.HasModelLayers = layersByGPU[g.ID]
		if !g.HasModelLayers && g.VRAMUsed > idleBaselineBytes {
			g.Ghost = true
		}
	}
}

// perGPUBreakdown returns per-device weights/cache/graph. Multi-GPU layouts
// keep device-level attribution so callers can answer "how much weight memory
// is on GPU 0" — the reason this endpoint exists.
func perGPUBreakdown(mem *ml.BackendMemory) map[string]*VRAMBreakdown {
	if mem == nil || len(mem.GPUs) == 0 {
		return nil
	}
	out := make(map[string]*VRAMBreakdown, len(mem.GPUs))
	for _, g := range mem.GPUs {
		b := &VRAMBreakdown{Graph: g.Graph, Context: g.Context, Cublas: g.Cublas}
		for _, w := range g.Weights {
			b.Weights += w
		}
		for _, c := range g.Cache {
			b.KVCache += c
		}
		out[g.DeviceID.ID] = b
	}
	return out
}

func gpuComputeString(major, minor int) string {
	return strconv.Itoa(major) + "." + strconv.Itoa(minor)
}
