/**
 * Media for the image and audio prompts (#542), built at run time rather than
 * committed as fixtures: the same picture and words TC-MODELS-016 generated.
 * prompts.yaml names which (media: disc, or media: speech with its words).
 */
import { crc32, deflateSync } from 'node:zlib';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function pngChunk(type: string, data: Buffer): Buffer {
  const body = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body) >>> 0);
  return Buffer.concat([len, body, crc]);
}

/** 224x224 RGB PNG, base64: a white disc on black — a simple, describable shape. */
export function discPng(): string {
  const w = 224;
  const h = 224;
  const px = Buffer.alloc(h * (1 + w * 3));
  let o = 0;
  for (let y = 0; y < h; y++) {
    px[o++] = 0; // filter byte per row
    for (let x = 0; x < w; x++) {
      const v = Math.hypot(x - 112, y - 112) < 70 ? 255 : 0;
      px[o++] = v;
      px[o++] = v;
      px[o++] = v;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr.set([8, 2, 0, 0, 0], 8); // 8-bit, RGB, default compression/filter/interlace
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', deflateSync(px)),
    pngChunk('IEND', Buffer.alloc(0)),
  ]).toString('base64');
}

/** `words` spoken by espeak-ng, as a base64 WAV. Needs espeak-ng on the host. */
export async function spokenWav(words: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'ollama37-audio-'));
  try {
    const file = join(dir, 'speech.wav');
    await promisify(execFile)('espeak-ng', ['-s', '130', '-w', file, words]);
    return (await readFile(file)).toString('base64');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
