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
	var loadAll bool
	flag.StringVar(&modelPath, "model", "", "Path to model directory (containing config.json)")
	flag.StringVar(&shardPath, "shard", "", "Optional .safetensors shard to load + probe via cgo")
	flag.BoolVar(&loadAll, "load-weights", false, "Load all shards via index.json and probe a few tensors from different shards")
	flag.Parse()
	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen_runner -model PATH [-shard SHARD.safetensors] [-load-weights]")
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

	if loadAll {
		if shardPath == "" {
			// mlx.Init wasn't called above; do it now.
			if err := mlx.Init(); err != nil {
				fmt.Fprintf(os.Stderr, "qwen_runner: %v\n", err)
				os.Exit(5)
			}
		}
		w, err := qwen.LoadWeights(modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "qwen_runner: load-weights: %v\n", err)
			os.Exit(6)
		}
		defer w.Release()
		fmt.Printf("qwen_runner: weights loaded, %d tensors across %d shards\n",
			w.Count(), w.NumShards())

		// Probe tensors that live in different shards, to prove the
		// index routing works end-to-end (not just shard-1 lookup).
		probes := []string{
			"language_model.model.embed_tokens.weight",         // shard 1
			"language_model.model.layers.0.linear_attn.in_proj_qkv.weight", // shard 1 or 2
			"language_model.lm_head.weight",                    // shard 4
		}
		for _, name := range probes {
			a := w.Get(name)
			if a == nil {
				fmt.Printf("qwen_runner: %s NOT FOUND in index\n", name)
				continue
			}
			fmt.Printf("qwen_runner: %s  shard=%s dtype=%s shape=%v\n",
				name, w.ShardOf(name), a.DType(), a.Shape())
			a.Release()
		}
	}

	fmt.Println("qwen_runner: OK")
}
