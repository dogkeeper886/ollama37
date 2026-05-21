package server

import "testing"

// TestAttributeGPUUsage covers the per-GPU attribution derived in
// attributeGPUUsage: physical used vs model-accounted, the unaccounted gap, the
// has-layers flag, and ghost detection (usage above idle on a layerless GPU).
func TestAttributeGPUUsage(t *testing.T) {
	const mib = uint64(1024 * 1024)

	gpus := []GPUMetrics{
		{ID: "model-gpu", VRAMTotal: 12000 * mib, VRAMFree: 2000 * mib},  // used 10000, has layers
		{ID: "ghost-gpu", VRAMTotal: 12000 * mib, VRAMFree: 11900 * mib}, // used 100, no layers -> ghost
		{ID: "idle-gpu", VRAMTotal: 12000 * mib, VRAMFree: 11996 * mib},  // used 4, below idle baseline
	}
	models := []ModelMetrics{
		{GPUs: []string{"model-gpu"}, VRAMByGPU: map[string]uint64{"model-gpu": 9800 * mib}},
	}

	attributeGPUUsage(gpus, models)

	model, ghost, idle := gpus[0], gpus[1], gpus[2]

	// Model GPU: accounted, with a small unaccounted remainder; not a ghost.
	if !model.HasModelLayers || model.Ghost {
		t.Errorf("model-gpu HasModelLayers=%v Ghost=%v; want true/false", model.HasModelLayers, model.Ghost)
	}
	if model.VRAMUsed != 10000*mib || model.VRAMAccounted != 9800*mib || model.VRAMUnaccounted != 200*mib {
		t.Errorf("model-gpu used=%d accounted=%d unaccounted=%d; want %d/%d/%d",
			model.VRAMUsed, model.VRAMAccounted, model.VRAMUnaccounted, 10000*mib, 9800*mib, 200*mib)
	}

	// Ghost GPU: usage above idle, no layers -> flagged; all of it unaccounted.
	if ghost.HasModelLayers || !ghost.Ghost {
		t.Errorf("ghost-gpu HasModelLayers=%v Ghost=%v; want false/true", ghost.HasModelLayers, ghost.Ghost)
	}
	if ghost.VRAMUnaccounted != 100*mib {
		t.Errorf("ghost-gpu unaccounted=%d; want %d", ghost.VRAMUnaccounted, 100*mib)
	}

	// Idle GPU: usage below the baseline -> not a ghost.
	if idle.HasModelLayers || idle.Ghost {
		t.Errorf("idle-gpu HasModelLayers=%v Ghost=%v; want false/false", idle.HasModelLayers, idle.Ghost)
	}
}
