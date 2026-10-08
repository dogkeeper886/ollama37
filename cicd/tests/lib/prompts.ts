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
  /** Strings the reply must not contain (case-insensitive): refusals. */
  reject: string[];
  /** An image or audio clip to send with the prompt. */
  media?: Media;
  /** Tools offered to the model; each answers its fixed result. */
  tools: LocalTool[];
}

export type Media = { kind: 'disc' } | { kind: 'speech'; words: string };
export interface LocalTool { name: string; description: string; result: string }

interface Entry {
  text?: string; text_file?: string; task?: string;
  expect?: string; reject?: string[]; media?: Media; tools?: LocalTool[]; models?: string[];
  options?: Record<string, unknown>; judge?: string;
}
interface Judge { pass: 'yes' | 'no'; question: string }
interface File { prompts: Record<string, Entry>; judges: Record<string, Judge>; warmup: string }

const FILE = process.env.OLLAMA37_PROMPTS
  ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', 'prompts.yaml');

function load(): File {
  const f = parse(readFileSync(FILE, 'utf-8')) as File;
  const bad = (msg: string): never => { throw new Error(`${FILE}: ${msg}`); };
  if (!f?.prompts || !f?.judges) bad('needs `prompts` and `judges`');
  if (typeof f.warmup !== 'string' || !f.warmup.trim()) bad('needs `warmup`, the judge\'s throwaway first question');
  for (const [name, j] of Object.entries(f.judges)) {
    if (j?.pass !== 'yes' && j?.pass !== 'no') bad(`judge "${name}" needs pass: "yes" or "no"`);
    if (!j.question?.includes('{reply}')) bad(`judge "${name}" has no {reply}`);
  }
  for (const [name, e] of Object.entries(f.prompts)) {
    const sources = [e.text, e.text_file].filter((x) => x !== undefined).length;
    if (sources !== 1) bad(`prompt "${name}" needs exactly one of text, text_file`);
    if (!e.judge) bad(`prompt "${name}" names no judge`);
    if (!f.judges[e.judge!]) bad(`prompt "${name}" names judge "${e.judge}", which is not under judges`);
    if (e.media && !(e.media.kind === 'disc' || (e.media.kind === 'speech' && typeof e.media.words === 'string' && e.media.words))) {
      bad(`prompt "${name}": media must be {kind: disc} or {kind: speech, words: "..."}`);
    }
    for (const t of e.tools ?? []) if (!t.name || typeof t.result !== 'string') bad(`prompt "${name}": each tool needs a name and a result`);
    if (e.tools?.length && e.judge !== 'grounded') bad(`prompt "${name}" offers tools, so its judge must be grounded`);
  }
  return f;
}

const config = load();

/** The judge's throwaway first question. */
export const warmupQuestion = (): string => config.warmup;

/** All prompt names, for a script's --help and for validating its choice. */
export const promptNames = (): string[] => Object.keys(config.prompts);

/** Prompts whose `models` list names `model`: the model-specific paths models.ts adds. */
export const promptsFor = (model: string): string[] =>
  Object.entries(config.prompts).filter(([, e]) => e.models?.includes(model)).map(([n]) => n);

/** The prompt called `name`. */
export function prompt(name: string): Prompt {
  const e = config.prompts[name];
  if (!e) throw new Error(`no prompt "${name}" in ${FILE}; have: ${promptNames().join(', ')}`);
  const body = e.text_file ? readFileSync(resolve(dirname(FILE), e.text_file), 'utf-8').trimEnd() : e.text!;
  const text = e.task ? `${body}\n\n${e.task}` : body;
  return {
    name,
    text,
    options: { ...(e.options ?? {}) },
    judge: config.judges[e.judge!].question,
    judgeName: e.judge!,
    judgePass: config.judges[e.judge!].pass,
    expect: e.expect,
    reject: e.reject ?? [],
    media: e.media,
    tools: e.tools ?? [],
  };
}
