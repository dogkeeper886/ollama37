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

const chunkSize = 64

// triTypeLower corresponds to GGML_TRI_TYPE_LOWER
const triTypeLower = 3

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

func (o *Options) ssmHeadVDim() int {
	if o.ssmDtRank > 0 {
		return o.ssmDInner / o.ssmDtRank
	}
	return 0
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
	WQkv     *nn.Linear `gguf:"attn_qkv"`
	WQkvGate *nn.Linear `gguf:"attn_qkv_gate"`
	SSMAlpha *nn.Linear `gguf:"ssm_alpha"`
	SSMBeta  *nn.Linear `gguf:"ssm_beta"`
	SSMDt    ml.Tensor  // bias vector for decay
	SSMA     ml.Tensor  // decay weight matrix
	SSMConv  ml.Tensor  // 1D conv kernel
	SSMNorm  *nn.RMSNorm `gguf:"ssm_norm"`
	SSMOut   *nn.Linear  `gguf:"ssm_out"`
}

// Forward for DeltaNet layer - autoregressive single-token path
// This is the most common path during generation (one token at a time)
func (dn *DeltaNet) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	nTokens := hiddenStates.Dim(1)
	headKDim := opts.ssmDState
	numKHeads := opts.ssmNGroup
	numVHeads := opts.ssmDtRank
	headVDim := opts.ssmHeadVDim()

	// Input projections
	qkvMixed := dn.WQkv.Forward(ctx, hiddenStates)
	z := dn.WQkvGate.Forward(ctx, hiddenStates)

	beta := dn.SSMBeta.Forward(ctx, hiddenStates)
	alpha := dn.SSMAlpha.Forward(ctx, hiddenStates)

	// Decay computation: softplus(alpha + dt_bias) * ssm_a
	alphaBiased := alpha.Add(ctx, dn.SSMDt)
	alphaSoftplus := alphaBiased.Softplus(ctx)
	gate := alphaSoftplus.Mul(ctx, dn.SSMA)

	// TODO: convolution state management and SSM conv
	// For now, apply SSM conv directly
	convOutput := qkvMixed.SSMConv(ctx, dn.SSMConv)
	convOutput = convOutput.SILU(ctx)

	// Extract Q, K, V from conv output
	qkvDim := headKDim*numKHeads*2 + headVDim*numVHeads
	_ = qkvDim

	// Q: first portion
	qConv := convOutput.View(ctx, 0, headKDim*numKHeads, nTokens)
	qConv = qConv.Reshape(ctx, headKDim, numKHeads, nTokens)

	// K: second portion
	kConv := convOutput.View(ctx, headKDim*numKHeads*4, headKDim*numKHeads, nTokens)
	kConv = kConv.Reshape(ctx, headKDim, numKHeads, nTokens)

	// V: third portion
	vConv := convOutput.View(ctx, 2*headKDim*numKHeads*4, headVDim*numVHeads, nTokens)
	vConv = vConv.Reshape(ctx, headVDim, numVHeads, nTokens)

	// Repeat Q/K if head counts differ
	if numKHeads != numVHeads {
		qConv = qConv.Repeat4D(ctx, headKDim, numVHeads, nTokens, 1)
		kConv = kConv.Repeat4D(ctx, headKDim, numVHeads, nTokens, 1)
	}

	// L2 normalize Q and K
	qConv = qConv.L2Norm(ctx, opts.eps)
	kConv = kConv.L2Norm(ctx, opts.eps)

	// Scale Q
	scale := 1.0 / math.Sqrt(float64(headVDim))
	qConv = qConv.Scale(ctx, float64(scale))

	// Sigmoid beta
	beta = beta.Sigmoid(ctx)

	// TODO: DeltaNet state update (requires recurrent state cache)
	// For single token decode:
	// 1. state = state * exp(gate)
	// 2. kv_mem = (state * k).sum(-2)
	// 3. delta = (v - kv_mem) * beta
	// 4. state = state + k * delta
	// 5. output = (state * q).sum(-2)

	// Gated normalization: rms_norm(output) * silu(z)
	output := qConv // placeholder until state management is implemented
	output2d := output.Reshape(ctx, headVDim, numVHeads*nTokens)
	z2d := z.Reshape(ctx, headVDim, numVHeads*nTokens)

	normalized := dn.SSMNorm.Forward(ctx, output2d, opts.eps)
	gatedSilu := z2d.SILU(ctx)
	attnOut := normalized.Mul(ctx, gatedSilu)

	finalOutput := attnOut.Reshape(ctx, headVDim*numVHeads, nTokens)
	return dn.SSMOut.Forward(ctx, finalOutput)
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
	PostAttnNorm  *nn.RMSNorm `gguf:"attn_post_norm"`

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
