/**
 * LogParser - Extracts structured memory profile data from container logs.
 *
 * Parses two log formats:
 * - New Ollama engine (device.go): "model weights", "kv cache", "compute graph", "total memory"
 * - Old llama.cpp engine: "offloaded N/M layers", "compute buffer size", "graph nodes"
 */

export interface DeviceMemory {
  device: string;
  size: string;
}

export interface MemoryProfile {
  /** "ollama" or "llama.cpp" */
  engine: string;
  /** e.g. "27/27" or "64/65" */
  layerOffload: string;
  /** Per-device layer assignments, e.g. "CUDA0:12, CUDA1:12, CUDA3:3" */
  layerSplit: string;
  /** Per-device weight sizes */
  weights: DeviceMemory[];
  /** Per-device KV cache sizes */
  kvCache: DeviceMemory[];
  /** Per-device compute graph sizes */
  computeGraph: DeviceMemory[];
  /** Total memory used */
  totalMemory: string;
  /** Graph node counts (if available) */
  graphNodes: string;
}

/**
 * Parse container logs and extract memory profile.
 * Returns null if no memory data found.
 */
export function parseMemoryProfile(logs: string): MemoryProfile | null {
  if (!logs) return null;

  const lines = logs.split('\n');

  const weights: DeviceMemory[] = [];
  const kvCache: DeviceMemory[] = [];
  const computeGraph: DeviceMemory[] = [];
  let totalMemory = '';
  let layerOffload = '';
  let layerSplit = '';
  let graphNodes = '';
  let engine = '';

  for (const line of lines) {
    // New engine: device.go log lines
    // msg="model weights" device=CUDA0 size="797.5 MiB"
    const weightsMatch = line.match(/msg="model weights"\s+device=(\S+)\s+size="([^"]+)"/);
    if (weightsMatch) {
      engine = 'ollama';
      weights.push({ device: weightsMatch[1], size: weightsMatch[2] });
      continue;
    }

    // msg="kv cache" device=CUDA0 size="192.0 MiB"
    const kvMatch = line.match(/msg="kv cache"\s+device=(\S+)\s+size="([^"]+)"/);
    if (kvMatch) {
      kvCache.push({ device: kvMatch[1], size: kvMatch[2] });
      continue;
    }

    // msg="compute graph" device=CUDA0 size="411.8 MiB"
    const graphMatch = line.match(/msg="compute graph"\s+device=(\S+)\s+size="([^"]+)"/);
    if (graphMatch) {
      computeGraph.push({ device: graphMatch[1], size: graphMatch[2] });
      continue;
    }

    // msg="total memory" size="13.3 GiB"
    const totalMatch = line.match(/msg="total memory"\s+size="([^"]+)"/);
    if (totalMatch) {
      totalMemory = totalMatch[1];
      continue;
    }

    // New engine: offload lines from ggml.go
    // msg="offloaded 27/27 layers to GPU"
    const offloadMatch = line.match(/offloaded (\d+\/\d+) layers to GPU/);
    if (offloadMatch) {
      engine = engine || 'ollama';
      layerOffload = offloadMatch[1];
      continue;
    }

    // New engine: layer split from commit line
    // operation=commit layers="27[ID:GPU-xxx Layers:12(0..11) ID:GPU-yyy Layers:12(12..23)]"
    const splitMatch = line.match(/operation=commit\s+layers="(\d+)\[([^\]]+)\]"/);
    if (splitMatch) {
      const parts = splitMatch[2].match(/ID:GPU-\S+\s+Layers:(\d+)\(([^)]+)\)/g);
      if (parts) {
        const splits: string[] = [];
        parts.forEach((part, idx) => {
          const m = part.match(/Layers:(\d+)\(([^)]+)\)/);
          if (m) splits.push(`GPU${idx}:${m[1]}`);
        });
        layerSplit = splits.join(', ');
      }
      continue;
    }

    // Old engine: llama.cpp offload
    // load_tensors: offloaded 64/65 layers to GPU
    const oldOffloadMatch = line.match(/load_tensors:\s+offloaded (\d+\/\d+) layers to GPU/);
    if (oldOffloadMatch) {
      engine = 'llama.cpp';
      layerOffload = oldOffloadMatch[1];
      continue;
    }

    // Old engine: per-device buffer sizes
    // load_tensors: CUDA0 model buffer size = 3492.48 MiB
    const oldBufferMatch = line.match(/load_tensors:\s+(\S+) model buffer size\s+=\s+(\S+ \S+)/);
    if (oldBufferMatch) {
      weights.push({ device: oldBufferMatch[1], size: oldBufferMatch[2] });
      continue;
    }

    // Old engine: compute buffer size
    // llama_context: CUDA0 compute buffer size = 411.75 MiB
    const oldComputeMatch = line.match(/llama_context:.*?(\S+) compute buffer size\s+=\s+(\S+ \S+)/);
    if (oldComputeMatch) {
      computeGraph.push({ device: oldComputeMatch[1], size: oldComputeMatch[2] });
      continue;
    }

    // Old engine: graph nodes
    // llama_context: graph nodes = 5678
    const nodesMatch = line.match(/graph nodes\s+=\s+(\d+(?:\s*\(.*?\))?)/);
    if (nodesMatch) {
      graphNodes = nodesMatch[1];
      continue;
    }

    // Old engine: layer split from offload log
    // layers.split="[16 16 16 16]"
    const oldSplitMatch = line.match(/layers\.split="\[([^\]]+)\]"/);
    if (oldSplitMatch) {
      const counts = oldSplitMatch[1].trim().split(/\s+/);
      layerSplit = counts.map((c, i) => `GPU${i}:${c}`).join(', ');
      continue;
    }
  }

  // No data found
  if (!engine && weights.length === 0 && computeGraph.length === 0) {
    return null;
  }

  return {
    engine: engine || 'unknown',
    layerOffload,
    layerSplit,
    weights,
    kvCache,
    computeGraph,
    totalMemory,
    graphNodes,
  };
}

/**
 * Format a MemoryProfile as human-readable lines for console output.
 */
export function formatMemoryProfile(profile: MemoryProfile): string[] {
  const lines: string[] = [];

  lines.push(`Engine: ${profile.engine}`);

  if (profile.layerOffload) {
    lines.push(`Layers: ${profile.layerOffload} offloaded to GPU`);
  }
  if (profile.layerSplit) {
    lines.push(`Split: ${profile.layerSplit}`);
  }

  if (profile.weights.length > 0) {
    lines.push(`Weights: ${profile.weights.map(w => `${w.device}=${w.size}`).join(', ')}`);
  }
  if (profile.kvCache.length > 0) {
    lines.push(`KV cache: ${profile.kvCache.map(k => `${k.device}=${k.size}`).join(', ')}`);
  }
  if (profile.computeGraph.length > 0) {
    lines.push(`Compute: ${profile.computeGraph.map(g => `${g.device}=${g.size}`).join(', ')}`);
  }
  if (profile.totalMemory) {
    lines.push(`Total: ${profile.totalMemory}`);
  }
  if (profile.graphNodes) {
    lines.push(`Graph nodes: ${profile.graphNodes}`);
  }

  return lines;
}
