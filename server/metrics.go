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
	Weights uint64 `json:"weights"`  // model weights on GPU
	KVCache uint64 `json:"kv_cache"` // K/V attention cache
	Graph   uint64 `json:"graph"`    // scratch compute buffer
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

	// Collect device info from the first available runner. GetDeviceInfos
	// does IPC to the runner subprocess, so cap it with a short timeout —
	// a stuck runner must not hang the metrics endpoint.
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	var devices []ml.DeviceInfo
	for _, r := range runners {
		if r.llama == nil {
			continue
		}
		if infos := r.llama.GetDeviceInfos(ctx); len(infos) > 0 {
			devices = infos
			break
		}
	}

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
		gpus = append(gpus, GPUMetrics{
			ID:          d.DeviceID.ID,
			Name:        d.Name,
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

// perGPUBreakdown returns per-device weights/cache/graph. Multi-GPU layouts
// keep device-level attribution so callers can answer "how much weight memory
// is on GPU 0" — the reason this endpoint exists.
func perGPUBreakdown(mem *ml.BackendMemory) map[string]*VRAMBreakdown {
	if mem == nil || len(mem.GPUs) == 0 {
		return nil
	}
	out := make(map[string]*VRAMBreakdown, len(mem.GPUs))
	for _, g := range mem.GPUs {
		b := &VRAMBreakdown{Graph: g.Graph}
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
