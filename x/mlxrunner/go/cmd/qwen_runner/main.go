// qwen_runner — entry point for the Go-side Qwen3.6-35B-A3B-4bit runner
// on K80 (Phase D.3, #189).
//
// Phase D.3 step 1: prints the model config so the package layout +
// JSON parsing are validated independently of the cgo / MLX surface.
// Subsequent steps will add weight loading and forward-pass dispatch.
package main

import (
	"flag"
	"fmt"
	"os"

	qwen "github.com/dogkeeper886/ollama37/x/mlxrunner/models/qwen3_6_a3b"
)

func main() {
	var modelPath string
	flag.StringVar(&modelPath, "model", "", "Path to model directory (containing config.json)")
	flag.Parse()
	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen_runner -model PATH")
		os.Exit(2)
	}

	cfg, err := qwen.LoadConfig(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "qwen_runner: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("qwen_runner: architectures = %v\n", cfg.Architectures)
	fmt.Printf("qwen_runner: model_type = %s\n", cfg.ModelType)
	fmt.Printf("qwen_runner: tie_word_embeddings = %v\n", cfg.TieWordEmbeddings)
	fmt.Printf("qwen_runner: text.model_type = %s\n", cfg.TextConfig.ModelType)
	fmt.Printf("qwen_runner: text.num_hidden_layers = %d\n", cfg.TextConfig.NumHiddenLayers)
	fmt.Printf("qwen_runner: text.hidden_size = %d\n", cfg.TextConfig.HiddenSize)
	fmt.Printf("qwen_runner: text.num_attention_heads = %d\n", cfg.TextConfig.NumAttentionHeads)
	fmt.Printf("qwen_runner: text.num_key_value_heads = %d\n", cfg.TextConfig.NumKeyValueHeads)
	fmt.Printf("qwen_runner: text.vocab_size = %d\n", cfg.TextConfig.VocabSize)
	fmt.Printf("qwen_runner: text.num_experts = %d\n", cfg.TextConfig.NumExperts)
	fmt.Printf("qwen_runner: text.num_experts_per_tok = %d\n", cfg.TextConfig.NumExpertsPerTok)
	fmt.Printf("qwen_runner: quant = bits=%d group_size=%d mode=%s\n",
		cfg.Quantization.Bits, cfg.Quantization.GroupSize, cfg.Quantization.Mode)
	fmt.Println("qwen_runner: OK")
}
