# Scope: gemma4ua encoder-free audio for gemma4:12b (#374 follow-on)

**Status:** scoped, not implemented. Branch `issue-367-text-only-gemma4` (text + vision done).

## The correction that makes this feasible

An earlier conclusion ("audio needs a conformer → not feasible for gemma4:12b") was **wrong**.
Tracing the audio encoder across upstreams (mainline llama.cpp `tools/mtmd`, ollama Go
`model/models/gemma4`) found **two different gemma4 audio paths**:

| | Go new engine (`model_audio.go`) | Mainline llama.cpp `gemma4ua` |
|---|---|---|
| Pipeline | WAV → **mel spectrogram (FFT+filterbank)** → **conformer** (SSCP conv + attention blocks) | **encoder-free**: raw waveform → 640-sample frames → projection |
| Weights | `a.conv1d.*`, `a.blk.*` (conformer) | only `mm.a.input_projection.weight` |
| Variant | non-unified audio (gemma4a / audio build) | **gemma4ua (unified)** |

**gemma4:12b ships `gemma4ua`** (the projector blob declares it; the only audio tensor present is
`mm.a.input_projection.weight`). Mainline `clip.cpp` says it plainly:
`// GEMMA4UA is encoder-free: it uses n_mel_bins as a raw-waveform frame size (640) and has no FFT/filterbank`.
So the one weight we have **is all gemma4ua needs** → feasible, as a *shallow* port (simpler than the
vision embedder).

## Reference (mainline llama.cpp, verified)

- **hparams** (`clip.cpp` GEMMA4UA case): `audio_sample_rate=16000`, `eps=1e-6`, `n_mel_bins=640`
  (re-used as the **frame size**, NOT a mel-bin count), encoder-free (no FFT).
- **Preprocessor** (`mtmd-audio.cpp mtmd_audio_preprocessor_gemma4ua`): no FFT/filterbank;
  `frame_size=640` (640 samples @16kHz = 40 ms = 1 token); `n_tokens = ceil(n_samples/640)`;
  zero-pad the last frame; store frame-major so the tensor loads `[n_tokens, 640]`.
- **Graph** (`clip_graph_gemma4ua::build`): `inp = build_inp_raw(1)` → `permute(1,0,2,3)` (→`[640, n_tokens]`
  so the norm is over the frame) → `ggml_rms_norm(eps)` → `build_mm(mm_input_proj_w)`. Done.
- **Markers** (`mtmd.cpp`): `aud_beg="<|audio>"`, `aud_end="<audio|>"` (gemma4 `<|x>...<x|>` style — NOT
  whisper's `<|audio_bos|>`). Cross-check `model/models/gemma4/model.go`: audio=`<|audio>`, audio_end=`<audio|>`.
- **non-causal**: mainline includes the audio chunk in `mtmd_decode_use_non_causal` — confirm gemma4ua audio
  tokens want bidirectional attention (likely yes, mirroring vision).
- **n_output_tokens**: audio token count = `n_tokens` from the preprocessor (ceil(n_samples/640)).

## Our fork already has the audio plumbing

`llama/llama.cpp/tools/mtmd/` has `mtmd-audio.{cpp,h}`, `mtmd_audio_tokens`, `aud_beg`/`aud_end`,
and a whisper encoder (`build_whisper_enc`) for ULTRAVOX/QWEN2A. gemma4ua hangs onto this — but the
fork is the **older monolithic clip.cpp** (pre clip_graph_* refactor), so mainline's classes must be
adapted inline (a `build_gemma4ua()` method + a gemma4ua branch in the audio preprocessor), not copied.

## Implementation steps (file-by-file)

1. **`clip_init` (clip.cpp):** reverse the gemma4ua audio-**skip** — create + `load_hparams` +
   `load_tensors` + `alloc_compute_meta` the audio ctx (only `mm.a.input_projection` to load).
2. **gemma4ua hparams case (clip.cpp):** sample_rate 16000, eps 1e-6, n_mel_bins 640, encoder-free.
3. **Audio preprocessor (mtmd-audio.cpp):** add the 640-frame, no-FFT path (gate the whisper mel on
   `proj_type != GEMMA4UA`, mirroring mainline's `fft_based` flag).
4. **`build_gemma4ua()` (clip.cpp):** permute → rms_norm → build_mm (pattern off `build_whisper_enc`,
   far simpler); wire into the encode dispatch + `clip_n_output_tokens` (n_tokens) + the audio
   position/input fill.
5. **Markers (mtmd.cpp):** add the gemma4ua `aud_beg="<|audio>"`, `aud_end="<audio|>"` case.
6. **non-causal (mtmd.cpp):** add gemma4ua to `mtmd_decode_use_non_causal` if audio is bidirectional.
7. Remove the audio-skip log + the temporary vision debug dump.

## Verification

Feed a 16 kHz WAV via `/api/chat` with an audio attachment; confirm the runner loads the audio ctx
(no skip), encodes (n_tokens ~ duration/40ms), and the model responds about the audio content.

## Risks / open questions

- **Does an encoder-free raw-frame projection give *meaningful* audio understanding?** A 640-sample
  frame → single linear projection is very shallow; mainline implements it, so the model is trained for
  it, but quality is unverified on the K80. The vision min-40-token study is a separate follow-up.
- **Fork vs mainline mtmd structure**: the older clip.cpp lacks the clip_graph_*/preprocessor class
  split, so the adaptation is the bulk of the effort (the math is trivial).
- ~~**Audio input path end-to-end**~~ **RESOLVED — no API/runner/server changes needed.**
  `mtmd_helper_bitmap_init_from_buf` (our fork, mtmd-helper.cpp:410) **auto-detects audio**: if the
  buffer `is_audio_file` (WAV) it decodes to PCM → `mtmd_bitmap_init_from_audio`, else treats it as an
  image. The runner's `MultimodalTokenize` (llama.go:712) already calls it on the `images`-field bytes,
  so a **WAV base64'd into the existing `images` API field** flows straight through to the audio
  encoder. The whole audio framework (WAV decode, audio chunks, whisper encoder) is already present.
  So the port is PURELY the clip.cpp gemma4ua pieces (steps 1-7) — no Go/API/server work.
  (The model must report it supports audio: today we skip the audio ctx, so it errors "does not
  support audio input"; loading the gemma4ua audio ctx fixes that.)
