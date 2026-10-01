/**
 * VRAM the server under test holds, per die (#542).
 *
 * Whole-die usage would count another tenant's allocation on a shared host, so
 * rows are scoped to the server: the PIDs of its container (OLLAMA37_CONTAINER,
 * default ollama37), or -- when no such container runs, as on a host serving from
 * a local binary -- processes named ollama (weaker: a second ollama counts in).
 * Returns nothing on a host without nvidia-smi.
 */
import { execFileSync } from 'node:child_process';

const CONTAINER = process.env.OLLAMA37_CONTAINER || 'ollama37';
const run = (cmd: string, args: string[]) => execFileSync(cmd, args, { encoding: 'utf-8', stdio: ['ignore', 'pipe', 'pipe'] });

export interface DieUsage { dies: number; totalMib: number }

export function serverVram(): DieUsage {
  let rows: string[][];
  try {
    rows = run('nvidia-smi', ['--query-compute-apps=gpu_uuid,pid,used_memory,process_name', '--format=csv,noheader,nounits'])
      .split('\n').map((l) => l.split(',').map((c) => c.trim())).filter((c) => c.length === 4);
  } catch {
    return { dies: 0, totalMib: 0 };
  }
  let pids: Set<string> | undefined;
  try {
    pids = new Set(run('docker', ['top', CONTAINER, '-eo', 'pid']).split('\n').slice(1).map((l) => l.trim()).filter(Boolean));
  } catch {
    pids = undefined; // no such container: fall back to the process name
  }
  const mine = rows.filter((c) => (pids ? pids.has(c[1]) : /ollama/.test(c[3])));
  return { dies: new Set(mine.map((c) => c[0])).size, totalMib: mine.reduce((s, c) => s + Number(c[2]), 0) };
}
