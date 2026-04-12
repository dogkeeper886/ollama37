package qwen35

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Options struct {
	hiddenSize int
	numHeads   int
	numKVHeads int
	keyLength  int
	valueLength int

	// DeltaNet SSM parameters
	ssmDInner int
	ssmDState int
	ssmNGroup int
	ssmDtRank int

	eps       float32
	ropeBase  float32
	ropeScale float32

	// Per-layer head_count_kv array: 0 = DeltaNet, >0 = attention
	kvHeadsPerLayer []int
}

func (o *Options) isRecurrent(il int) bool {
	if il < len(o.kvHeadsPerLayer) {
		return o.kvHeadsPerLayer[il] == 0
	}
	return false
}

func (o *Options) headDim() int {
	if o.keyLength > 0 {
		return o.keyLength
	}
	if o.valueLength > 0 {
		return o.valueLength
	}
	return o.hiddenSize / o.numHeads
}

// Attention layer (standard grouped-query attention with RoPE)
type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options, il int) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	kvHeads := opts.kvHeadsPerLayer[il]

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	key = key.Reshape(ctx, opts.headDim(), kvHeads, batchSize)
	value = value.Reshape(ctx, opts.headDim(), kvHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	ropeOpts := []func(*rope.Options){rope.WithTypeNeoX()}
	query = fast.RoPE(ctx, query, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, ropeOpts...)
	key = fast.RoPE(ctx, key, positions, opts.headDim(), opts.ropeBase, 1./opts.ropeScale, ropeOpts...)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim())), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

// DeltaNet layer (linear attention with recurrent state)
type DeltaNet struct {
	WQkv     *nn.Linear  `gguf:"attn_qkv"`
	WQkvGate *nn.Linear  `gguf:"attn_gate"`
	SSMAlpha *nn.Linear  `gguf:"ssm_alpha"`
	SSMBeta  *nn.Linear  `gguf:"ssm_beta"`
	SSMDt    ml.Tensor   `gguf:"ssm_dt"`
	SSMA     ml.Tensor   `gguf:"ssm_a"`
	SSMConv1D ml.Tensor  `gguf:"ssm_conv1d"`
	SSMNorm  *nn.RMSNorm `gguf:"ssm_norm"`
	SSMOut   *nn.Linear  `gguf:"ssm_out"`
}

// Forward for DeltaNet layer
// TODO: Full DeltaNet implementation with recurrent state, conv1d, chunked attention.
// Currently a stub that passes through the output projection to maintain valid shapes
// for memory measurement. This allows the ollama engine's iterative Load loop to
// correctly estimate GPU count, even though inference output is not correct.
func (dn *DeltaNet) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	// Use output projection to produce correct output shape (hiddenSize, nTokens)
	// Input: (hiddenSize, nTokens), SSMOut weight: (hiddenSize, d_inner)
	// We need an intermediate of size (d_inner, nTokens)
	// Use WQkvGate which projects hiddenSize -> d_inner
	intermediate := dn.WQkvGate.Forward(ctx, hiddenStates)

	// Gated normalization stub: just use SSMNorm on intermediate
	normalized := dn.SSMNorm.Forward(ctx, intermediate, opts.eps)

	return dn.SSMOut.Forward(ctx, normalized)
}

// FFN (feed-forward network)
type FFN struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (ffn *FFN) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	return ffn.Down.Forward(ctx, ffn.Gate.Forward(ctx, hiddenStates).SILU(ctx, ffn.Up.Forward(ctx, hiddenStates)))
}

// Layer can be either attention or DeltaNet
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	PostAttnNorm  *nn.RMSNorm `gguf:"post_attention_norm"`

	// Only one of these is used per layer
	Attention *Attention
	DeltaNet  *DeltaNet

	FFN *FFN
}

func (l *Layer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options, il int) ml.Tensor {
	residual := hiddenStates

	// Pre-attention norm
	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	// Attention or DeltaNet
	if opts.isRecurrent(il) {
		hiddenStates = l.DeltaNet.Forward(ctx, hiddenStates, opts)
	} else {
		hiddenStates = l.Attention.Forward(ctx, hiddenStates, positions, cache, opts, il)
	}

	// Output filtering on last layer
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	// Residual connection
	hiddenStates = hiddenStates.Add(ctx, residual)

	// FFN with pre-norm and residual
	ffnResidual := hiddenStates
	hiddenStates = l.PostAttnNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = l.FFN.Forward(ctx, hiddenStates)
	return hiddenStates.Add(ctx, ffnResidual)
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []Layer `gguf:"blk"`

	*Options
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, m.Options, i)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeOpts := []func(*rope.Options){rope.WithTypeNeoX()}
	return fast.RoPE(ctx, key, shift, m.headDim(), m.ropeBase, 1./m.ropeScale, ropeOpts...), nil
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	blockCount := int(c.Uint("block_count"))

	// Read per-layer head_count_kv to determine which layers are DeltaNet vs attention
	kvHeadsRaw := c.Ints("attention.head_count_kv")
	kvHeadsPerLayer := make([]int, blockCount)
	if len(kvHeadsRaw) >= blockCount {
		for i := 0; i < blockCount; i++ {
			kvHeadsPerLayer[i] = int(kvHeadsRaw[i])
		}
	} else if len(kvHeadsRaw) == 1 {
		// scalar: all layers have same kv heads
		for i := range kvHeadsPerLayer {
			kvHeadsPerLayer[i] = int(kvHeadsRaw[0])
		}
	}

	layers := make([]Layer, blockCount)
	for i := range layers {
		if kvHeadsPerLayer[i] == 0 {
			// DeltaNet layer
			layers[i].DeltaNet = &DeltaNet{}
		} else {
			// Attention layer
			layers[i].Attention = &Attention{}
		}
		layers[i].FFN = &FFN{}
	}

	opts := &Options{
		hiddenSize:      int(c.Uint("embedding_length")),
		numHeads:        int(c.Uint("attention.head_count")),
		numKVHeads:      int(c.Uint("attention.head_count_kv")),
		keyLength:       int(c.Uint("attention.key_length")),
		valueLength:     int(c.Uint("attention.value_length")),
		ssmDInner:       int(c.Uint("ssm.inner_size")),
		ssmDState:       int(c.Uint("ssm.state_size")),
		ssmNGroup:       int(c.Uint("ssm.group_count")),
		ssmDtRank:       int(c.Uint("ssm.dt_rank")),
		eps:             c.Float("attention.layer_norm_rms_epsilon"),
		ropeBase:        c.Float("rope.freq_base"),
		ropeScale:       c.Float("rope.scaling.factor", 1),
		kvHeadsPerLayer: kvHeadsPerLayer,
	}

	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers:  layers,
		Options: opts,
	}

	// For now, use CausalCache for the attention layers
	// TODO: implement HybridCache with recurrent state for DeltaNet layers
	m.Cache = kvcache.NewCausalCache(m.Shift)

	return &m, nil
}

func init() {
	model.Register("qwen35", New)
}
