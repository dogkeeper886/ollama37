/**
 * The Ollama client, on a socket without undici's deadline.
 *
 * Node's global fetch imposes its own ~300s headers/body timeout that
 * AbortSignal cannot lift. With `stream: false` Ollama sends no headers until
 * generation finishes, so any K80 request that runs longer than five minutes —
 * a 31b model at num_predict 400, or a cold registry pull — is killed
 * mid-flight. node:http has no such cap, so the caller's timeout is the only
 * deadline. Everything above the socket stays the library's.
 */
import http from 'node:http';
import https from 'node:https';
import { Ollama, type Fetch } from 'ollama';

/** The YAML testcases allowed up to 20 minutes per step; nothing here may be stricter. */
export const DEFAULT_TIMEOUT_MS = 1_200_000;

export function nodeFetch(timeoutMs: number): Fetch {
  return (input, init) =>
    new Promise<Response>((resolve, reject) => {
      const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url);
      const transport = url.protocol === 'https:' ? https : http;
      const body = init?.body as string | undefined;

      const headers: Record<string, string> = {};
      new Headers(init?.headers).forEach((v, k) => (headers[k] = v));
      if (body !== undefined) headers['content-length'] = String(Buffer.byteLength(body));

      const req = transport.request(url, { method: init?.method ?? 'GET', headers }, (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        // The response is a separate emitter from req — a reset mid-body (a real risk on a
        // slow/OOMing K80 backend) errors here, not on req. Without this it's an uncaught throw.
        res.on('error', reject);
        res.on('end', () =>
          resolve(
            new Response(Buffer.concat(chunks), {
              status: res.statusCode ?? 502,
              statusText: res.statusMessage ?? '',
              // node lowercases header names and gives set-cookie as an array; the client
              // reads content-type only, so keep the plain string ones.
              headers: Object.fromEntries(
                Object.entries(res.headers).filter((e): e is [string, string] => typeof e[1] === 'string'),
              ),
            }),
          ),
        );
      });

      const signal = AbortSignal.timeout(timeoutMs);
      const onAbort = () => req.destroy(new Error(`ollama request timed out after ${timeoutMs}ms`));
      signal.addEventListener('abort', onAbort, { once: true });
      req.on('close', () => signal.removeEventListener('abort', onAbort));
      req.on('error', reject);
      req.end(body);
    });
}

/** An Ollama client that will wait as long as a K80 needs. */
export const ollamaClient = (host: string, timeoutMs = DEFAULT_TIMEOUT_MS): Ollama =>
  new Ollama({ host, fetch: nodeFetch(timeoutMs) });
