/**
 * A fake Ollama for the canary (#542): answers every prompt with one fixed reply.
 *
 *   npx tsx canary/mock-ollama.ts <port> good|salad
 *
 * `salad` is fluent word salad -- real words, no exact repetition, so the script
 * check passes it and only the judge can catch it. `good` is a sound short answer
 * the judge must pass. Load, unload and listing calls answer empty.
 */
import http from 'node:http';

const REPLIES: Record<string, string> = {
  good: 'Two plus two equals four.',
  salad: 'Purple the quickly tomorrow of because elephant singing never table if the under running cloud seventeen whereas lamp government softly beneath it was and blue therefore spoon election window perhaps river among the eaten',
};
const [port, mode] = [Number(process.argv[2]), process.argv[3]];
const reply = REPLIES[mode];
if (!port || !reply) { console.error('usage: mock-ollama.ts <port> good|salad'); process.exit(2); }

const send = (res: http.ServerResponse, obj: unknown) => {
  const body = JSON.stringify(obj);
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
};

http.createServer((req, res) => {
  let raw = '';
  req.on('data', (c) => (raw += c));
  req.on('end', () => {
    const body = raw ? JSON.parse(raw) : {};
    if (req.method === 'GET') return send(res, { models: [] });
    if (req.url === '/api/generate' && body.prompt) {
      return send(res, { model: body.model, response: reply, done: true, done_reason: 'stop',
        prompt_eval_count: 20, prompt_eval_duration: 1e8, eval_count: 12, eval_duration: 1e9 });
    }
    send(res, { model: body.model ?? '', response: '', done: true });
  });
}).listen(port, '127.0.0.1');
