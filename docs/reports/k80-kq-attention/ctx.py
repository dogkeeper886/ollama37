# Decode tok/s against prompt length on ollama.service. Usage: python3 -I ctx.py <label> <model>...
import json, sys, urllib.error, urllib.request
def gen(model, prompt, ctx, n, retry=True):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                       "options": {"num_predict": n, "temperature": 0, "num_ctx": ctx}}).encode()
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", body), timeout=1800))
    except urllib.error.HTTPError as e:
        print(f"  {model} HTTP {e.code}: {e.read()[:200]!r}" + ("; retrying" if retry else ""), flush=True)
        if not retry:
            raise
        return gen(model, prompt, ctx, n, False)
filler = "The lighthouse keeper wrote in his log that the sea was calm and the lamp was clean. "
label = sys.argv[1]
for model in sys.argv[2:]:
    for words, ctx in [(0, 8192), (3000, 8192), (6500, 16384)]:
        p = filler * (words // 17) + "Summarize the log in one sentence."
        gen(model, p, ctx, 1)
        d = gen(model, p, ctx, 64)
        tps = d["eval_count"] / d["eval_duration"] * 1e9
        print(f"{label} {model:14} prompt {d['prompt_eval_count']:5d}  decode {tps:6.2f} tok/s  | {' '.join(d['response'].split())[:90]}", flush=True)
    urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", json.dumps({"model": model, "keep_alive": 0}).encode())).read()
