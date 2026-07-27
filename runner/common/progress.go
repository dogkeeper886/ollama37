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
// The shape is ported from llama.cpp's server, which prints the same three kinds
// of line (tools/server/server-context.cpp: print_timings_pp, print_timings_tg,
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
	// A prompt ingested faster than this reports no progress at all — only the
	// summary. Anything slow enough to be worth watching crosses it.
	progressMinPrefill = 3 * time.Second

	// Smallest gap between two progress lines for the same sequence. Without it
	// a long prompt would emit a line per batch.
	progressInterval = 3 * time.Second

	// Decode stays quiet until the rate means something.
	progressMinDecoded = 100
)

// Prefill reports how far through its prompt a sequence is. tokens must count
// only the tokens actually pushed through a batch — a prefix served from the KV
// cache was never computed, and including it reports a rate the hardware cannot
// reach. remaining is what is still to ingest.
func (p *Progress) Prefill(tokens, remaining int, elapsed time.Duration) {
	if elapsed < progressMinPrefill || !p.prefill.due(elapsed, tokens) {
		return
	}

	total := tokens + remaining
	slog.Info("prefill progress",
		"tokens", tokens,
		"total", total,
		"progress", round(float64(tokens)/float64(total)),
		"elapsed", elapsed.Round(time.Millisecond),
		"tps", rate(tokens, elapsed))
}

// Decode reports generation speed while it is still generating. It carries a
// recent-window rate alongside the lifetime one: a model slowing down as its KV
// cache fills is visible in the window long before it moves the average.
func (p *Progress) Decode(tokens int, elapsed time.Duration) {
	if tokens < progressMinDecoded {
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
		"tokens", tokens,
		"elapsed", elapsed.Round(time.Millisecond),
		"tps", rate(tokens, elapsed),
		"tps_recent", recent)
}

// Summary logs a request's totals for both phases. Unthrottled: exactly one line
// per sequence, whatever ended it — including a client that hung up mid-
// generation, which is the case a caller-side log would miss.
func Summary(reason string, prefillTokens int, prefillElapsed time.Duration, decodeTokens int, decodeElapsed time.Duration) {
	if reason == "" {
		// llm.DoneReason.String() has no text for a closed connection.
		reason = "closed"
	}

	slog.Info("completion",
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
