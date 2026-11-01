package server

import "testing"

func TestAttributeGPUUsage(t *testing.T) {
	const mib = uint64(1024 * 1024)
	tests := []struct {
		name          string
		gpus          []GPUMetrics
		models        []ModelMetrics
		wantUsed      []uint64
		wantAccounted []uint64
		wantUnaccount []uint64
		wantLayers    []bool
		wantGhost     []bool
	}{
		{
			// Idle server: the engine opens a CUDA context per device. That is
			// expected usage, not a ghost (#98/#138), so the flag stays off.
			name: "idle discovery contexts are not ghosts",
			gpus: []GPUMetrics{
				{ID: "0", VRAMTotal: 11441 * mib, VRAMFree: (11441 - 112) * mib},
				{ID: "1", VRAMTotal: 11441 * mib, VRAMFree: (11441 - 112) * mib},
			},
			wantUsed:      []uint64{112 * mib, 112 * mib},
			wantAccounted: []uint64{0, 0},
			wantUnaccount: []uint64{112 * mib, 112 * mib},
			wantLayers:    []bool{false, false},
			wantGhost:     []bool{false, false},
		},
		{
			// One model on GPU0 while GPU1 holds more than a bare context: the
			// historical ghost signature, flagged for investigation.
			name: "residual usage on a device without layers is a ghost",
			gpus: []GPUMetrics{
				{ID: "0", VRAMTotal: 11441 * mib, VRAMFree: 11441 * mib},
				{ID: "1", VRAMTotal: 11441 * mib, VRAMFree: (11441 - 512) * mib},
			},
			models: []ModelMetrics{
				{Name: "m", GPUs: []string{"0"}, VRAMByGPU: map[string]uint64{"0": 6000 * mib}},
			},
			wantUsed:      []uint64{0, 512 * mib},
			wantAccounted: []uint64{6000 * mib, 0},
			wantUnaccount: []uint64{0, 512 * mib},
			wantLayers:    []bool{true, false},
			wantGhost:     []bool{false, true},
		},
		{
			// The serving device holds layers: never a ghost, even with
			// unaccounted headroom (contexts, fragmentation).
			name: "device with layers is never a ghost",
			gpus: []GPUMetrics{
				{ID: "0", VRAMTotal: 11441 * mib, VRAMFree: (11441 - 8000) * mib},
			},
			models: []ModelMetrics{
				{Name: "m", GPUs: []string{"0"}, VRAMByGPU: map[string]uint64{"0": 7000 * mib}},
			},
			wantUsed:      []uint64{8000 * mib},
			wantAccounted: []uint64{7000 * mib},
			wantUnaccount: []uint64{1000 * mib},
			wantLayers:    []bool{true},
			wantGhost:     []bool{false},
		},
		{
			// Driver quirks can report free > total; VRAMUsed must stay 0
			// rather than underflow uint64.
			name: "free above total does not underflow",
			gpus: []GPUMetrics{
				{ID: "0", VRAMTotal: 100 * mib, VRAMFree: 200 * mib},
			},
			models: []ModelMetrics{
				{Name: "m", GPUs: []string{"0"}, VRAMByGPU: map[string]uint64{"0": 300 * mib}},
			},
			wantUsed:      []uint64{0},
			wantAccounted: []uint64{300 * mib},
			wantUnaccount: []uint64{0},
			wantLayers:    []bool{true},
			wantGhost:     []bool{false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpus := append([]GPUMetrics(nil), tt.gpus...)
			attributeGPUUsage(gpus, tt.models)
			for i, g := range gpus {
				if g.VRAMUsed != tt.wantUsed[i] {
					t.Errorf("gpu %s VRAMUsed = %d, want %d", g.ID, g.VRAMUsed, tt.wantUsed[i])
				}
				if g.VRAMAccounted != tt.wantAccounted[i] {
					t.Errorf("gpu %s VRAMAccounted = %d, want %d", g.ID, g.VRAMAccounted, tt.wantAccounted[i])
				}
				if g.VRAMUnaccounted != tt.wantUnaccount[i] {
					t.Errorf("gpu %s VRAMUnaccounted = %d, want %d", g.ID, g.VRAMUnaccounted, tt.wantUnaccount[i])
				}
				if g.HasModelLayers != tt.wantLayers[i] {
					t.Errorf("gpu %s HasModelLayers = %v, want %v", g.ID, g.HasModelLayers, tt.wantLayers[i])
				}
				if g.Ghost != tt.wantGhost[i] {
					t.Errorf("gpu %s Ghost = %v, want %v", g.ID, g.Ghost, tt.wantGhost[i])
				}
			}
		})
	}
}
