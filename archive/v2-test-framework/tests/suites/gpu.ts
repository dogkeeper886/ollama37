/**
 * VRAM attributed to the Ollama server's own processes.
 *
 * `nvidia-smi --query-gpu=memory.used` is whole-die usage, which on a shared
 * testbed counts another tenant's allocation as this model's footprint. The YAML
 * testcases scoped it by intersecting compute-apps rows with the container's
 * PIDs (`docker top $OLLAMA37_CONTAINER`); this keeps that.
 */
import { execa } from 'execa';

const CONTAINER = process.env.OLLAMA37_CONTAINER;

export interface ComputeApp {
  uuid: string;
  pid: string;
  usedMib: number;
}

/** PIDs inside the server's container; undefined when it runs on the host. */
async function serverPids(): Promise<Set<string> | undefined> {
  if (!CONTAINER) return undefined;
  const { stdout } = await execa('docker', ['top', CONTAINER, '-eo', 'pid']);
  return new Set(
    stdout
      .split('\n')
      .slice(1)
      .map((l) => l.trim())
      .filter(Boolean),
  );
}

/**
 * Compute-apps rows belonging to the server: its container's PIDs where there is
 * a container, else processes named ollama — which is weaker, since a second
 * ollama on the box would be counted in.
 */
export async function serverComputeApps(): Promise<ComputeApp[]> {
  const pids = await serverPids();
  const { stdout } = await execa('nvidia-smi', [
    '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
    '--format=csv,noheader,nounits',
  ]);
  return stdout
    .split('\n')
    .map((l) => l.split(',').map((c) => c.trim()))
    .filter((c) => c.length === 4 && (pids ? pids.has(c[1]) : /ollama/.test(c[3])))
    .map((c) => ({ uuid: c[0], pid: c[1], usedMib: Number(c[2]) }));
}
