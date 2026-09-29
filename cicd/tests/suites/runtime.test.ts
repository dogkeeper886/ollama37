/**
 * The container: does it start with the GPU passed through, does it report
 * healthy, does it see CUDA, and does /api/metrics answer in its documented
 * shape (#130)?
 *
 * Replaces testcases/runtime/TC-RUNTIME-*.yml (#535). Starting the container is
 * opt-in: the old TC-RUNTIME-001 ran `docker compose down` unconditionally, so
 * running any runtime testcase tore down whatever was serving.
 */
import { describe, test, expect, beforeAll } from 'vitest';
import { execa } from 'execa';
import { flag } from './gates.js';

const ROOT = process.env.OLLAMA37_ROOT ?? `${process.cwd()}/../..`;
const CONTAINER = process.env.OLLAMA37_CONTAINER ?? 'ollama37';
const HOST = process.env.OLLAMA_HOST ?? 'http://localhost:11434';
const RESTART = flag('OLLAMA37_RESTART_CONTAINER');

const compose = (args: string[]) => execa('docker', ['compose', ...args], { cwd: `${ROOT}/docker` });
const inContainer = (args: string[]) => execa('docker', ['exec', CONTAINER, ...args]);

/** Container health, or 'not_found' when it isn't running. */
async function health(): Promise<string> {
  const { stdout } = await execa('docker', ['inspect', CONTAINER, '--format', '{{.State.Health.Status}}'], {
    reject: false,
  });
  return stdout.trim() || 'not_found';
}

describe('container', () => {
  test.skipIf(!RESTART)('starts under compose', async ({ annotate }) => {
    await compose(['down']).catch(() => {});
    await compose(['up', '-d']);
    for (let i = 0; i < 30; i++) {
      if ((await health()) === 'healthy') break;
      await new Promise((r) => setTimeout(r, 2000));
    }
    const { stdout } = await compose(['ps']);
    await annotate(stdout.split('\n').slice(1).join('\n'));
    expect(stdout).toContain('Up');
  });

  test('reports healthy', async ({ annotate }) => {
    const status = await health();
    await annotate(status);
    expect(status).toBe('healthy');
  });

  test('runs an ollama binary', async ({ annotate }) => {
    const { stdout } = await inContainer(['ollama', '--version']);
    await annotate(stdout.trim());
    expect(stdout).toMatch(/ollama/);
  });
});

describe('GPU passthrough', () => {
  test('nvidia-smi sees the devices', async ({ annotate }) => {
    const { stdout } = await inContainer(['nvidia-smi']);
    await annotate(stdout.split('\n').find((l) => l.includes('CUDA Version')) ?? '');
    expect(stdout).toContain('CUDA Version');
    expect(stdout).not.toContain('NVIDIA-SMI has failed');
    expect(stdout).not.toContain('No devices were found');
  });

  test('the CUDA libraries are on the loader path', async () => {
    const { stdout } = await execa('bash', ['-c', `docker exec ${CONTAINER} ldconfig -p | grep -i cuda`]);
    expect(stdout).toMatch(/cuda/i);
  });

  // Unified memory needs its device node. It appears on first CUDA use, so a
  // freshly booted host may not have one until something asks for it.
  test('the UVM device exists', async () => {
    const present = async () => (await execa('test', ['-e', '/dev/nvidia-uvm'], { reject: false })).exitCode === 0;
    if (!(await present())) {
      await execa('sudo', ['nvidia-modprobe', '-u', '-c=0'], { reject: false });
    }
    expect(await present()).toBe(true);
  });
});

describe('/api/metrics', () => {
  let body: Record<string, unknown>;
  let status: number;

  let raw: string;

  beforeAll(async () => {
    const res = await fetch(`${HOST}/api/metrics`);
    status = res.status;
    // Read as text and parse after the status assertion: an image without this
    // endpoint answers HTML, and parsing first fails every test in this block
    // with a syntax error instead of reporting the status.
    raw = await res.text();
    try {
      body = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      body = {};
    }
  });

  test('answers 200', () => {
    expect(status, raw.slice(0, 200)).toBe(200);
  });

  test('carries gpus, models, errors and totals', () => {
    expect(Array.isArray(body.gpus)).toBe(true);
    expect(Array.isArray(body.models)).toBe(true);
    expect(body.errors).toBeTypeOf('object');
    expect(body.totals).toBeTypeOf('object');
  });

  test('describes at least one GPU', async ({ annotate }) => {
    // beforeAll swallows a parse failure into {} so the status is reported rather
    // than a syntax error — which leaves body.gpus undefined here.
    expect(Array.isArray(body.gpus), raw.slice(0, 200)).toBe(true);
    const gpus = body.gpus as Array<{ id: unknown; name: unknown; vram_total: unknown }>;
    expect(gpus.length).toBeGreaterThanOrEqual(1);
    await annotate(`${gpus.length} GPU(s), first: ${String(gpus[0]?.name)}`);
    expect(gpus[0].id).toBeTypeOf('string');
    expect(gpus[0].name).toBeTypeOf('string');
    expect(gpus[0].vram_total).toBeTypeOf('number');
    expect(gpus[0].vram_total as number).toBeGreaterThan(0);
  });
});

describe('API', () => {
  test('/api/tags lists models', async ({ annotate }) => {
    const res = await fetch(`${HOST}/api/tags`);
    const tags = (await res.json()) as { models?: unknown[] };
    await annotate(`${tags.models?.length ?? 0} model(s)`);
    expect(Array.isArray(tags.models)).toBe(true);
  });
});
