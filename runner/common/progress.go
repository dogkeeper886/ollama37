package common

import (
	"log/slog"
	"math"
	"time"
)

// Progress emits the prefill and decode progress lines for one sequence, and
// throttles them. A runner's log is the container log, so these are what an
// operator watching `docker logs` sees while a request is still running — on a
// K80 that can be minutes of otherwise total silence.
//
// The shape is ported from llama.cpp's server, which prints the same kinds of
// line (tools/server/server-context.cpp: print_timings_pp, print_timings_tg,
// print_timings). We can't inherit them: this fork vendors llama.cpp without
// tools/server and serves inference from its own runners.
//
// Both throttles measure the phase's own accumulated duration rather than wall
// time. That keeps the interval on the same clock as the rate being reported,
// and keeps this type free of any dependency on the current time.
type Progress struct {
	prefill phase
	decode  phase
}

// phase is the throttle state for one of the two phases. They are kept apart
// because their clocks are unrelated — decode elapsed starts back at zero, so
// sharing a "last printed at" between them would suppress decode entirely.
type phase struct {
	printed     bool
	lastElapsed time.Duration
	lastTokens  int
}

const (
	// A phase faster than this says nothing — only the summary reports it. Anything
	// slow enough to be worth watching crosses it.
	//
	// llama.cpp additionally holds decode back for its first 100 tokens, because on
	// the hardware it targets 100 tokens can pass in under a second. Here that floor
	// would buy silence, not quiet: at a K80's few tokens per second it is ~16 s of
	// nothing after the prompt lands, and a request answering in under 100 tokens
	// would never report a rate at all. The interval below already does the job.
	progressMinElapsed = 3 * time.Second

	// Smallest gap between two progress lines for the same sequence. Without it a
	// long prompt would emit a line per batch.
	progressInterval = 3 * time.Second
)

// Prefill reports how far through its prompt a sequence is. tokens must count
// only the tokens actually pushed through a batch — a prefix served from the KV
// cache was never computed, and including it reports a rate the hardware cannot
// reach. remaining is what is still to ingest.
//
// Note that tokens and remaining therefore describe the *uncached* work, not the
// prompt: on a warm multi-turn request they sum to far less than the prompt the
// caller sent. Neither is named as a prompt size for that reason.
//
// Call this only from a pass that actually placed tokens in the batch. A sequence
// shut out of a batch is still charged the time, and reporting it would re-print a
// stale count at a decaying rate — the caller has the per-pass count and this does
// not, so the check cannot live here.
//
// elapsed is that sequence's share of shared batch time, not hardware time: every
// live sequence is charged the whole batch, so with several prefilling at once each
// one's tps reads low. It is the same duration the API reports for the request.
func (p *Progress) Prefill(seq, tokens, remaining int, elapsed time.Duration) {
	if elapsed < progressMinElapsed || !p.prefill.due(elapsed, tokens) {
		return
	}

	slog.Info("prefill progress",
		"seq", seq,
		"tokens", tokens,
		"remaining", remaining,
		"progress", round(float64(tokens)/float64(tokens+remaining)),
		"elapsed", elapsed.Round(time.Millisecond),
		"tps", rate(tokens, elapsed))
}

// PrefillDone reports the finished prompt. It closes the gap the throttled lines
// leave: the last of those is always a partial fraction, and decode then has its
// own quiet spell before it can report a rate, so without this an operator cannot
// tell a prompt that landed from one that stalled.
//
// For an embedding sequence this is the terminal line — there is no decode phase,
// and Summary is deliberately not called for one (a summary per embedding would
// flood a bulk indexing run, which the min-elapsed gate here already keeps quiet).
func (p *Progress) PrefillDone(seq, tokens int, elapsed time.Duration) {
	if elapsed < progressMinElapsed {
		return
	}

	slog.Info("prefill done",
		"seq", seq,
		"tokens", tokens,
		"elapsed", elapsed.Round(time.Millisecond),
		"tps", rate(tokens, elapsed))
}

// Decode reports generation speed while it is still generating. It carries a
// recent-window rate alongside the lifetime one: a model slowing down as its KV
// cache fills is visible in the window long before it moves the average.
func (p *Progress) Decode(seq, tokens int, elapsed time.Duration) {
	if elapsed < progressMinElapsed {
		return
	}

	// Measure the window before due() overwrites the marks it needs.
	recent := rate(tokens, elapsed)
	if p.decode.printed {
		recent = rate(tokens-p.decode.lastTokens, elapsed-p.decode.lastElapsed)
	}

	if !p.decode.due(elapsed, tokens) {
		return
	}

	slog.Info("decode progress",
		"seq", seq,
		"tokens", tokens,
		"elapsed", elapsed.Round(time.Millisecond),
		"tps", rate(tokens, elapsed),
		"tps_recent", recent)
}

// Summary logs a request's totals for both phases. Unthrottled: exactly one line
// per sequence, whatever ended it — including a client that hung up mid-
// generation, which is the case a caller-side log would miss.
//
// prefillTokens is the computed prefill, so it is below the API's
// prompt_eval_count whenever the cache served part of the prompt.
//
// Both durations are the ones the API reports for the same request, kept identical
// on purpose so the log and the API never disagree. That inherits their quirks:
// decodeElapsed covers one step fewer than decodeTokens counts (the first token's
// compute is charged to prefill), so a one-token generation reads
// decode_tokens=1 decode_elapsed=0s decode_tps=0, and a short one overstates the
// rate a little. Correcting it here would make the log contradict the API.
func Summary(seq int, reason string, prefillTokens int, prefillElapsed time.Duration, decodeTokens int, decodeElapsed time.Duration) {
	if reason == "" {
		// llm.DoneReason.String() has no text for a closed connection.
		reason = "closed"
	}

	slog.Info("completion",
		"seq", seq,
		"reason", reason,
		"prefill_tokens", prefillTokens,
		"prefill_elapsed", prefillElapsed.Round(time.Millisecond),
		"prefill_tps", rate(prefillTokens, prefillElapsed),
		"decode_tokens", decodeTokens,
		"decode_elapsed", decodeElapsed.Round(time.Millisecond),
		"decode_tps", rate(decodeTokens, decodeElapsed))
}

// due reports whether this phase may print now, and records the marks a later
// window measurement reads.
func (p *phase) due(elapsed time.Duration, tokens int) bool {
	if p.printed && elapsed-p.lastElapsed < progressInterval {
		return false
	}

	p.printed = true
	p.lastElapsed = elapsed
	p.lastTokens = tokens
	return true
}

func rate(tokens int, elapsed time.Duration) float64 {
	if elapsed <= 0 {
		return 0
	}

	return round(float64(tokens) / elapsed.Seconds())
}

func round(v float64) float64 {
	return math.Round(v*100) / 100
}
