/**
 * The images: does the builder carry the toolchain this fork needs, does the
 * runtime image build from source, does it ship without that toolchain, and does
 * it carry the tag compose reads?
 *
 * Replaces testcases/build/TC-BUILD-*.yml (#535). The one-hour compile is opt-in
 * here rather than a dependency edge: `--id TC-BUILD-004` used to drag
 * TC-BUILD-002 in behind it, and a test run that silently starts an hour of nvcc
 * is a surprise, not a dependency.
 */
import { describe, test, expect } from 'vitest';
import { execa } from 'execa';
import { flag } from './gates.js';

const ROOT = process.env.OLLAMA37_ROOT ?? `${process.cwd()}/../..`;
const BUILD_IMAGE = flag('OLLAMA37_BUILD_IMAGE');

const docker = (args: string[]) => execa('docker', args);

describe('builder image', () => {
  test('exists', async () => {
    const { stdout } = await docker(['images', 'ollama37-builder:latest', '--format', '{{.Repository}}:{{.Tag}}']);
    expect(stdout).toContain('ollama37-builder:latest');
  });

  // The fork's whole constraint chain: sm37 needs driver 470, which caps CUDA at
  // 11.4, which needs GCC 10. A builder that drifted off any of these produces
  // an image that cannot target compute 3.7.
  test('carries CUDA 11.4', async ({ annotate }) => {
    const { stdout } = await docker(['run', '--rm', 'ollama37-builder:latest', 'nvcc', '--version']);
    await annotate(stdout.split('\n').at(-1) ?? '');
    expect(stdout).toContain('Cuda compilation tools');
    expect(stdout).toMatch(/release 11\.4/);
  });

  test('carries GCC 10', async ({ annotate }) => {
    const { stdout } = await docker(['run', '--rm', 'ollama37-builder:latest', 'gcc', '--version']);
    await annotate(stdout.split('\n')[0]);
    expect(stdout).toMatch(/gcc.*10/);
  });

  test('carries Go 1.20+', async ({ annotate }) => {
    const { stdout } = await docker(['run', '--rm', 'ollama37-builder:latest', 'go', 'version']);
    await annotate(stdout.trim());
    expect(stdout).toMatch(/go1\.2[0-9]/);
  });
});

describe('runtime image', () => {
  test.skipIf(!BUILD_IMAGE)(
    'builds from local source',
    async () => {
      // The YAML ran this as `2>&1`, so its reject patterns saw BuildKit's output,
      // which docker writes to stderr. `all` is execa's merged stream.
      const { all } = await execa('make', ['build-runtime-local-no-cache'], {
        cwd: `${ROOT}/docker`,
        env: { OLLAMA_VERSION: process.env.OLLAMA_VERSION ?? '0.0.0' },
        all: true,
      });
      expect(all).toContain('Runtime image built successfully');
      expect(all).not.toMatch(/[Ee]rror:/);
    },
    3_600_000,
  );

  test('is tagged ollama37:latest', async () => {
    const { stdout } = await docker(['images', 'ollama37:latest', '--format', '{{.Repository}}:{{.Tag}} {{.Size}}']);
    expect(stdout).toContain('ollama37:latest');
  });

  test('ships without the toolchain it was built with', async ({ annotate }) => {
    const { stdout } = await docker(['images', 'ollama37:latest', '--format', '{{.Size}}']);
    const { stdout: bytes } = await execa('numfmt', ['--from=iec', stdout.trim().replace(/B$/, '')]);
    const gb = Number(bytes) / 1073741824;
    await annotate(`${gb.toFixed(2)} GB`);
    expect(gb).toBeGreaterThanOrEqual(1);
    expect(gb).toBeLessThanOrEqual(3);
  });

  // Only behind a build. TC-BUILD-004 depended on TC-BUILD-002, so the retag
  // never ran on its own; retagging a stale local ollama37:latest as the compose
  // image makes the next `compose up` serve the old binary.
  test.skipIf(!BUILD_IMAGE)('carries the tag compose reads', async ({ annotate }) => {
    await docker(['tag', 'ollama37:latest', 'dogkeeper886/ollama37:latest']);
    const { stdout } = await docker(['image', 'inspect', 'dogkeeper886/ollama37:latest', '--format', '{{.Id}}']);
    await annotate(stdout.trim());
    expect(stdout).toMatch(/^sha256:/);
  });
});
