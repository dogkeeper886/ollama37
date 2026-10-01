/**
 * The only reader of prompts.yaml (#542).
 *
 * Every prompt a test sends and every question the judge asks lives in that one
 * file. This loads it once, rejects a broken file before any model is called, and
 * hands out a prompt by name -- so a test never holds prompt text, and each prompt
 * brings its own judge question.
 *
 * OLLAMA37_PROMPTS points at another file for an experiment.
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';

export interface Prompt {
  name: string;
  /** The text sent to the model. */
  text: string;
  /** Ollama request options. */
  options: Record<string, unknown>;
  /** The judge question with `{reply}` (and `{result}` for grounded) still in it. */
  judge: string;
  judgeName: string;
  /** The judge's answer that passes the reply. */
  judgePass: 'yes' | 'no';
  /** A string the reply must contain, when the prompt plants one. */
  expect?: string;
}

interface Long { filler: string; sentence: string; needle: string; depth: number }
interface Entry {
  text?: string; text_file?: string; task?: string; long?: Long;
  expect?: string; options?: Record<string, unknown>; judge?: string;
}
interface Judge { pass: 'yes' | 'no'; question: string }
interface File { prompts: Record<string, Entry>; judges: Record<string, Judge> }

const FILE = process.env.OLLAMA37_PROMPTS
  ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', 'prompts.yaml');

function load(): File {
  const f = parse(readFileSync(FILE, 'utf-8')) as File;
  const bad = (msg: string): never => { throw new Error(`${FILE}: ${msg}`); };
  if (!f?.prompts || !f?.judges) bad('needs `prompts` and `judges`');
  for (const [name, j] of Object.entries(f.judges)) {
    if (j?.pass !== 'yes' && j?.pass !== 'no') bad(`judge "${name}" needs pass: "yes" or "no"`);
    if (!j.question?.includes('{reply}')) bad(`judge "${name}" has no {reply}`);
  }
  for (const [name, e] of Object.entries(f.prompts)) {
    const sources = [e.text, e.text_file, e.long].filter((x) => x !== undefined).length;
    if (sources !== 1) bad(`prompt "${name}" needs exactly one of text, text_file, long`);
    if (!e.judge) bad(`prompt "${name}" names no judge`);
    if (!f.judges[e.judge!]) bad(`prompt "${name}" names judge "${e.judge}", which is not under judges`);
    if (e.long && !(e.long.depth > 0 && e.long.depth < 1)) bad(`prompt "${name}": long.depth must be in (0, 1)`);
  }
  return f;
}

const config = load();

/**
 * Filler to about `tokens` tokens, the needle at `depth`, the task last. The same
 * target gives the same bytes, so two runs time the same prompt. ~0.8 words per
 * token for this filler; each sentence is 11 words.
 */
function buildLong(l: Long, tokens: number): string {
  const words = l.filler.split(/\s+/).filter(Boolean);
  const targetWords = Math.max(64, Math.floor(tokens * 0.8));
  const needleAt = Math.floor(targetWords * l.depth);
  const parts: string[] = [];
  let i = 0;
  let placed = false;
  for (let n = 0; n < targetWords; n += 11) {
    if (!placed && n >= needleAt) { parts.push(l.needle); placed = true; }
    const chunk: string[] = [];
    for (let k = 0; k < 9; k++) { chunk.push(words[(i + 7 * k) % words.length]); i += 63; }
    parts.push(l.sentence.replace('{words}', chunk.join(' ')).replace('{word}', words[i % words.length]));
    i += 1;
  }
  if (!placed) parts.push(l.needle);
  return parts.join(' ');
}

/** All prompt names, for a script's --help and for validating its choice. */
export const promptNames = (): string[] => Object.keys(config.prompts);

/** The prompt called `name`. `tokens` sizes a `long` prompt and is ignored otherwise. */
export function prompt(name: string, tokens?: number): Prompt {
  const e = config.prompts[name];
  if (!e) throw new Error(`no prompt "${name}" in ${FILE}; have: ${promptNames().join(', ')}`);
  let body: string;
  if (e.long) {
    if (!tokens) throw new Error(`prompt "${name}" is long: pass a token count`);
    body = buildLong(e.long, tokens);
  } else if (e.text_file) {
    body = readFileSync(resolve(dirname(FILE), e.text_file), 'utf-8').trimEnd();
  } else {
    body = e.text!;
  }
  const text = e.task ? `${body}\n\n${e.task}` : body;
  return {
    name,
    text,
    options: { ...(e.options ?? {}) },
    judge: config.judges[e.judge!].question,
    judgeName: e.judge!,
    judgePass: config.judges[e.judge!].pass,
    expect: e.expect,
  };
}
