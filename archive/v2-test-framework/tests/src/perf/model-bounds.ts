/**
 * Resolve a model's native context length + tool capability (STORY-022 #446).
 *
 * The fit-map sweep must not exceed a model's trained context (a longer window is
 * context-shifted, not a real measurement) and must pick a judge the model can satisfy
 * (a no-tools model can't drive an MCP tool call). Both facts come from `/api/show` —
 * the JSON `ollama show` reads — so the sweep bounds itself with no per-model setup.
 */
export interface ModelBounds {
  /** Trained context length from `<arch>.context_length`; 0 if the blob doesn't declare it. */
  nativeCtx: number;
  /** Model advertises the "tools" capability (can drive an MCP tool call). */
  tools: boolean;
  /** general.architecture, for the report. */
  arch: string;
}

export async function modelBounds(host: string, model: string): Promise<ModelBounds> {
  const res = await fetch(`${host}/api/show`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: model }),
  });
  if (!res.ok) throw new Error(`/api/show ${model}: HTTP ${res.status}`);
  const data: any = await res.json();
  const info = data.model_info ?? {};
  const arch = String(info['general.architecture'] ?? '');
  const nativeCtx = Number(info[`${arch}.context_length`] ?? 0) || 0;
  const tools = Array.isArray(data.capabilities) && data.capabilities.includes('tools');
  return { nativeCtx, tools, arch };
}
