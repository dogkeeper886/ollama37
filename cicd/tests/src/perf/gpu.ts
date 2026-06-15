/**
 * GPU introspection for the perf subcommands
 * (port of the get_gpu_info / get_gpu_offload helpers in benchmark-throughput.sh).
 *
 * gpuInfo() reads per-GPU memory via nvidia-smi; gpuOffload() asks ollama
 * /api/ps what fraction of a loaded model sits in VRAM. Both degrade to an
 * empty/zero result when the tool or endpoint is unavailable.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export interface GpuRow {
  index: number;
  name: string;
  usedMib: number;
  totalMib: number;
  freeMib: number;
}

/** Per-GPU memory via nvidia-smi; [] if nvidia-smi is absent or fails. */
export async function gpuInfo(): Promise<GpuRow[]> {
  try {
    const { stdout } = await execFileAsync('nvidia-smi', [
      '--query-gpu=index,name,memory.used,memory.total,memory.free',
      '--format=csv,noheader,nounits',
    ]);
    return stdout
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
      .map((line) => {
        const [index, name, used, total, free] = line.split(',').map((c) => c.trim());
        return {
          index: Number(index),
          name,
          usedMib: Number(used),
          totalMib: Number(total),
          freeMib: Number(free),
        };
      });
  } catch {
    return [];
  }
}

/** Percent of `model` resident in VRAM per ollama /api/ps; 0 if unknown. */
export async function gpuOffload(host: string, model: string): Promise<number> {
  try {
    const res = await fetch(`${host}/api/ps`);
    const ps: any = await res.json();
    const m = (ps.models ?? []).find((x: any) => x.name === model || x.model === model);
    if (m && m.size > 0) {
      return Math.round((m.size_vram / m.size) * 100);
    }
    return 0;
  } catch {
    return 0;
  }
}
