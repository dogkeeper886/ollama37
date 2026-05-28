// Phase D.2 (#189) — Go cgo runner for MLX on K80.
//
// Standalone module (separate from the main github.com/ollama/ollama
// go.mod) so the MLX runner can be developed without touching ollama's
// dependency graph until the API surface stabilizes. Will be folded
// into the main module or wired via replace directive once the runner
// is usable.

module github.com/dogkeeper886/ollama37/x/mlxrunner

go 1.24.0
