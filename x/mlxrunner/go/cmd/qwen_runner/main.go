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
	var loadAll, probeOps bool
	flag.StringVar(&modelPath, "model", "", "Path to model directory (containing config.json)")
	flag.StringVar(&shardPath, "shard", "", "Optional .safetensors shard to load + probe via cgo")
	flag.BoolVar(&loadAll, "load-weights", false, "Load all shards via index.json and probe a few tensors from different shards")
	flag.BoolVar(&probeOps, "probe-ops", false, "Run the qwen_load_go op chain against real weights (mirrors PR #210/#216 output). Implies -load-weights.")
	flag.Parse()
	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen_runner -model PATH [-shard SHARD.safetensors] [-load-weights] [-probe-ops]")
		os.Exit(2)
	}
	if probeOps {
		loadAll = true
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

		if probeOps {
			if err := runProbeOps(w); err != nil {
				fmt.Fprintf(os.Stderr, "qwen_runner: probe-ops: %v\n", err)
				os.Exit(7)
			}
		}
	}

	fmt.Println("qwen_runner: OK")
}

// runProbeOps replays qwen_load_go's full 5-stage chain (#216) on real
// layer-0 weights to validate the Go wrapper surface. Output should
// match qwen_load_go byte-for-byte; any drift means a wrapper bug.
//
// Stages:
//   1. take(embed, [42]) + dequantize     -> bf16 [1, 2048]
//   2. rms_norm(stage1, input_layernorm)  -> bf16 [1, 2048]
//   3. quantized_matmul(stage2, in_proj_qkv) -> bf16 [1, 8192]
//   4. conv1d(ones([1,4,8192]), layer0.conv1d.w, groups=8192) -> bf16 [1,1,8192]
//   5. silu(stage4) = sigmoid * x         -> bf16 [1, 1, 8192]
func runProbeOps(w *qwen.Weights) error {
	wq := w.Get("language_model.model.embed_tokens.weight")
	sc := w.Get("language_model.model.embed_tokens.scales")
	bs := w.Get("language_model.model.embed_tokens.biases")
	if wq == nil || sc == nil || bs == nil {
		return fmt.Errorf("missing embed_tokens tensors")
	}
	defer wq.Release()
	defer sc.Release()
	defer bs.Release()

	// Stage 1: take + dequant of row 42.
	idx, err := mlx.FromInt32([]int32{42})
	if err != nil {
		return err
	}
	defer idx.Release()
	wqRow, err := wq.Take(idx, 0)
	if err != nil {
		return err
	}
	defer wqRow.Release()
	scRow, err := sc.Take(idx, 0)
	if err != nil {
		return err
	}
	defer scRow.Release()
	bsRow, err := bs.Take(idx, 0)
	if err != nil {
		return err
	}
	defer bsRow.Release()
	full, err := mlx.Dequantize(wqRow, scRow, bsRow, 64, 4, "affine")
	if err != nil {
		return err
	}
	defer full.Release()
	if err := mlx.Eval(full); err != nil {
		return err
	}
	fmt.Printf("qwen_runner: dequantize(embed row 42)  dtype=%s shape=%v size=%d\n",
		full.DType(), full.Shape(), full.Size())
	if err := printFirst5(full, "dequant row 42"); err != nil {
		return err
	}

	// Stage 2: rms_norm with layer-0 input_layernorm gain.
	normGain := w.Get("language_model.model.layers.0.input_layernorm.weight")
	if normGain == nil {
		return fmt.Errorf("missing input_layernorm.weight")
	}
	defer normGain.Release()
	normed, err := mlx.RMSNorm(full, normGain, 1e-6)
	if err != nil {
		return err
	}
	defer normed.Release()
	if err := printFirst5(normed, "rms_norm(dequant row 42)"); err != nil {
		return err
	}
	absMean, err := scalarAbsMean(normed)
	if err != nil {
		return err
	}
	fmt.Printf("qwen_runner: rms_norm abs_mean=%g\n", absMean)

	// Stage 3: quantized_matmul through layer-0 in_proj_qkv.
	qkvW := w.Get("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
	qkvS := w.Get("language_model.model.layers.0.linear_attn.in_proj_qkv.scales")
	qkvB := w.Get("language_model.model.layers.0.linear_attn.in_proj_qkv.biases")
	if qkvW == nil || qkvS == nil || qkvB == nil {
		return fmt.Errorf("missing in_proj_qkv tensors")
	}
	defer qkvW.Release()
	defer qkvS.Release()
	defer qkvB.Release()
	qkv, err := mlx.QuantizedMatmul(normed, qkvW, qkvS, qkvB, true, 64, 4, "affine")
	if err != nil {
		return err
	}
	defer qkv.Release()
	fmt.Printf("qwen_runner: in_proj_qkv(rms_normed) shape=%v\n", qkv.Shape())
	if err := printFirst5(qkv, "in_proj_qkv"); err != nil {
		return err
	}

	// Stage 4: conv1d on synthetic ones input. Groups=8192 = depthwise.
	convW := w.Get("language_model.model.layers.0.linear_attn.conv1d.weight")
	if convW == nil {
		return fmt.Errorf("missing conv1d.weight")
	}
	defer convW.Release()
	convIn, err := mlx.Ones([]int64{1, 4, 8192}, "bfloat16")
	if err != nil {
		return err
	}
	defer convIn.Release()
	convOut, err := mlx.Conv1d(convIn, convW, 1, 0, 1, 8192)
	if err != nil {
		return err
	}
	defer convOut.Release()
	fmt.Printf("qwen_runner: conv1d(ones, layer_0.conv1d.w) shape=%v\n", convOut.Shape())
	if err := printFirst5(convOut, "conv1d"); err != nil {
		return err
	}

	// Stage 5: silu = sigmoid(x) * x.
	sig, err := convOut.Sigmoid()
	if err != nil {
		return err
	}
	defer sig.Release()
	silu, err := sig.Multiply(convOut)
	if err != nil {
		return err
	}
	defer silu.Release()
	if err := printFirst5(silu, "silu(conv1d)"); err != nil {
		return err
	}

	return nil
}

// printFirst5 astype-fp32s, evals, copies the first 5 values to host,
// and prints. Same shape as qwen_load_go's first5OfArray helper so
// the output lines line up for byte-for-byte comparison.
func printFirst5(arr *mlx.Array, label string) error {
	f32, err := arr.ToFloat32()
	if err != nil {
		return err
	}
	defer f32.Release()
	if err := mlx.Eval(f32); err != nil {
		return err
	}
	out := make([]float32, 5)
	n, err := f32.CopyFloat32(out)
	if err != nil {
		return err
	}
	fmt.Printf("qwen_runner: %s first %d = %v\n", label, n, out[:n])
	return nil
}

// scalarAbsMean evaluates abs(arr).mean() and extracts the scalar.
func scalarAbsMean(arr *mlx.Array) (float32, error) {
	a, err := arr.Abs()
	if err != nil {
		return 0, err
	}
	defer a.Release()
	m, err := a.MeanAll()
	if err != nil {
		return 0, err
	}
	defer m.Release()
	f32, err := m.ToFloat32()
	if err != nil {
		return 0, err
	}
	defer f32.Release()
	if err := mlx.Eval(f32); err != nil {
		return 0, err
	}
	out := make([]float32, 1)
	if _, err := f32.CopyFloat32(out); err != nil {
		return 0, err
	}
	return out[0], nil
}
