// Package museglimmer is Meta's Muse Glimmer (#481), ported from llama.cpp's
// src/models/muse-glimmer.cpp (b10969).
package museglimmer

import (
	"bytes"
	"image"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// The post-attention and post-FFN norms use their own epsilon.
const postNormEps = 1e-8

const (
	cacheTypeSWA = iota
	cacheTypeCausal
)

type Options struct {
	hiddenSize, numHeads, numKVHeads, headDim int
	eps                                       float32
	ropeBase                                  float32
	logitScale, finalLogitSoftcap             float32
	slidingWindowPattern                      []bool
}

// isSWA reports a sliding-window layer. Those use RoPE; the full-attention
// layers use no positional encoding.
func (o *Options) isSWA(layer int) bool {
	return layer < len(o.slidingWindowPattern) && o.slidingWindowPattern[layer]
}

type Model struct {
	model.Base
	model.BytePairEncoding

	*VisionModel `gguf:"v"`
	*Projector   `gguf:"mm"`
	ImageProcessor

	imageStartToken, imageEndToken, imagePatchToken int32

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		VisionModel:     newVisionModel(c),
		Projector:       &Projector{},
		ImageProcessor:  newImageProcessor(c),
		imageStartToken: -1,
		imageEndToken:   -1,
		imagePatchToken: -1,
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				// The glimmer renderer writes <|begin_of_text|> itself; adding it
				// here as the GGUF asks would send it twice.
				AddBOS: false,
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{
						int32(c.Uint("tokenizer.ggml.eos_token_id")),
						int32(c.Uint("tokenizer.ggml.eot_token_id")),
					},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			// tokenizer.ggml.pre "llama4"
			`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize:           int(c.Uint("embedding_length")),
			numHeads:             int(c.Uint("attention.head_count")),
			numKVHeads:           int(c.Uint("attention.head_count_kv")),
			headDim:              int(c.Uint("attention.key_length")),
			eps:                  c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:             c.Float("rope.freq_base"),
			logitScale:           c.Float("logit_scale", 1),
			finalLogitSoftcap:    c.Float("final_logit_softcapping"),
			slidingWindowPattern: c.Bools("attention.sliding_window_pattern"),
		},
	}

	for i, tok := range c.Strings("tokenizer.ggml.tokens") {
		switch tok {
		case "<|image_start|>":
			m.imageStartToken = int32(i)
		case "<|image_end|>":
			m.imageEndToken = int32(i)
		case "<|patch|>":
			m.imagePatchToken = int32(i)
		}
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWACache(int32(c.Uint("attention.sliding_window")), m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return &m, nil
}

var _ model.MultimodalProcessor = (*Model)(nil)

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if m.VisionModel == nil || len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixels, gridW, gridH := m.ImageProcessor.ProcessImage(img)
	pixelValues := ctx.Input().FromFloats(pixels, gridW*m.ImageProcessor.patchSize, gridH*m.ImageProcessor.patchSize, 3)

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, gridW, gridH)
	return []input.Multimodal{{Tensor: m.Projector.Forward(ctx, visionOutputs)}}, nil
}

// PostTokenize wraps each image's embeddings in <|image_start|> ... <|image_end|>,
// as llama.cpp's mtmd does for this projector.
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input
	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		numTokens := inp.Multimodal[0].Tensor.Dim(1)
		result = append(result,
			&input.Input{Token: m.imageStartToken},
			&input.Input{Token: m.imagePatchToken, Multimodal: inp.Multimodal, MultimodalHash: inp.MultimodalHash, SameBatch: numTokens},
		)
		result = append(result, slices.Repeat([]*input.Input{{Token: m.imagePatchToken}}, numTokens-1)...)
		result = append(result, &input.Input{Token: m.imageEndToken})
	}
	return result, nil
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Gate      *nn.Linear  `gguf:"attn_gate"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (sa *Attention) Forward(ctx ml.Context, layer int, hiddenState, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, opts.headDim, opts.numHeads, batchSize)
	q = sa.QueryNorm.Forward(ctx, q, opts.eps)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)
	k = sa.KeyNorm.Forward(ctx, k, opts.eps)

	if opts.isSWA(layer) {
		q = fast.RoPE(ctx, q, positions, opts.headDim, opts.ropeBase, 1)
		k = fast.RoPE(ctx, k, positions, opts.headDim, opts.ropeBase, 1)
	}

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, opts.headDim, opts.numKVHeads, batchSize)

	kqv := nn.Attention(ctx, q, k, v, 1/math.Sqrt(float64(opts.headDim)), cache)
	kqv = kqv.Reshape(ctx, opts.headDim*opts.numHeads, batchSize)

	// Output gate: sigmoid of a projection of the attention input, before o_proj.
	kqv = kqv.Mul(ctx, sa.Gate.Forward(ctx, hiddenState).Sigmoid(ctx))
	return sa.Output.Forward(ctx, kqv)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	if !m.isSWA(layer) {
		return key, nil
	}
	return fast.RoPE(ctx, key, shift, m.headDim, m.ropeBase, 1), nil
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention     *Attention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`
	MLPNorm           *nn.RMSNorm `gguf:"ffn_norm"`
	MLP               *MLP
	PostMLPNorm       *nn.RMSNorm `gguf:"post_ffw_norm"`
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenState, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, layer, hiddenState, positions, cache, opts)
	hiddenState = l.PostAttentionNorm.Forward(ctx, hiddenState, postNormEps)

	// In the final layer (outputs != nil), keep only the positions that need logits.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState)
	hiddenState = l.PostMLPNorm.Forward(ctx, hiddenState, postNormEps)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	// Image embeddings replace their placeholder tokens.
	if len(batch.Multimodal) > 0 {
		hiddenState = hiddenState.Duplicate(ctx)
		for _, image := range batch.Multimodal {
			visionOutputs := image.Multimodal[0].Tensor
			ctx.Forward(visionOutputs.Copy(ctx, hiddenState.View(ctx, image.Index*hiddenState.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))
		}
	}

	// An unweighted RMS norm on the embeddings.
	hiddenState = hiddenState.RMSNorm(ctx, nil, m.eps)

	for i, layer := range m.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
			cacheType := cacheTypeCausal
			if m.isSWA(i) {
				cacheType = cacheTypeSWA
			}
			m.Cache.(*kvcache.WrapperCache).SetLayerType(cacheType)
		}

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, outputs, m.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)
	hiddenState = hiddenState.Scale(ctx, float64(m.logitScale))

	if m.finalLogitSoftcap > 0 {
		hiddenState = hiddenState.Scale(ctx, 1/float64(m.finalLogitSoftcap))
		hiddenState = hiddenState.Tanh(ctx)
		hiddenState = hiddenState.Scale(ctx, float64(m.finalLogitSoftcap))
	}

	return hiddenState, nil
}

func init() {
	model.Register("muse-glimmer", New)
}
