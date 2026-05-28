// qwen_runner — entry point for the Go-side Qwen3.6-35B-A3B-4bit runner
// on K80 (Phase D.3, #189).
//
// Phase D.3 step 1: print the model config (independent of cgo).
// Phase D.3 step 2: optionally load a safetensors shard via the mlx
// Go wrapper and report tensor count. Subsequent steps add the
// multi-shard merge, layer dispatch, and forward pass.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/dogkeeper886/ollama37/x/mlxrunner/mlx"
	qwen "github.com/dogkeeper886/ollama37/x/mlxrunner/models/qwen3_6_a3b"
)

func main() {
	var modelPath, shardPath string
	flag.StringVar(&modelPath, "model", "", "Path to model directory (containing config.json)")
	flag.StringVar(&shardPath, "shard", "", "Optional .safetensors shard to load + probe via cgo")
	flag.Parse()
	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen_runner -model PATH [-shard SHARD.safetensors]")
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

	if shardPath != "" {
		if err := mlx.Init(); err != nil {
			fmt.Fprintf(os.Stderr, "qwen_runner: %v\n", err)
			os.Exit(3)
		}
		st, err := mlx.LoadSafetensors(shardPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "qwen_runner: %v\n", err)
			os.Exit(4)
		}
		defer st.Release()
		fmt.Printf("qwen_runner: shard %s loaded, %d tensors\n", shardPath, st.Count())

		// Spot-check a couple of known tensors. embed_tokens lives in
		// shard 1; the values mirror qwen_load_go's first lines so a
		// regression in the Go wrapper would surface here.
		probe := []string{
			"language_model.model.embed_tokens.weight",
			"language_model.model.embed_tokens.scales",
		}
		for _, name := range probe {
			a := st.Get(name)
			if a == nil {
				fmt.Printf("qwen_runner: %s NOT FOUND (skipping)\n", name)
				continue
			}
			fmt.Printf("qwen_runner: %s  dtype=%s shape=%v size=%d\n",
				name, a.DType(), a.Shape(), a.Size())
			a.Release()
		}

		// Negative-case probe: missing tensor returns nil.
		missing := "language_model.NOT_A_REAL_TENSOR"
		if a := st.Get(missing); a == nil {
			fmt.Printf("qwen_runner: %s correctly returned nil\n", missing)
		} else {
			fmt.Printf("qwen_runner: %s unexpectedly found, dtype=%s\n", missing, a.DType())
			a.Release()
		}
	}

	fmt.Println("qwen_runner: OK")
}
