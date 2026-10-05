package museglimmer

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// Every sparseFactor-th layer, and the last, attends across the whole image;
// the others attend within windows. Ported from llama.cpp's
// tools/mtmd/models/muse-glimmer.cpp and its clip.cpp inputs (b10969).
const sparseFactor = 4

type VisionOptions struct {
	hiddenSize, numHeads, headDim int
	patchSize, merge              int
	eps, ropeBase                 float32
}

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

// rope2D rotates the first half of each head by column and the second half by row.
func rope2D(ctx ml.Context, t, posW, posH ml.Tensor, opts *VisionOptions) ml.Tensor {
	half := opts.headDim / 2
	first := t.View(ctx, 0, half, t.Stride(1), opts.numHeads, t.Stride(2), t.Dim(2))
	first = nn.RoPE(ctx, first, posW, half, opts.ropeBase, 1)
	second := t.View(ctx, half*t.Stride(0), half, t.Stride(1), opts.numHeads, t.Stride(2), t.Dim(2))
	second = nn.RoPE(ctx, second, posH, half, opts.ropeBase, 1)
	return first.Concat(ctx, second, 0)
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenState, posW, posH, mask ml.Tensor, opts *VisionOptions) ml.Tensor {
	numPatches := hiddenState.Dim(1)

	q := sa.Query.Forward(ctx, hiddenState).Reshape(ctx, opts.headDim, opts.numHeads, numPatches)
	k := sa.Key.Forward(ctx, hiddenState).Reshape(ctx, opts.headDim, opts.numHeads, numPatches)
	v := sa.Value.Forward(ctx, hiddenState).Reshape(ctx, opts.headDim, opts.numHeads, numPatches)

	q = rope2D(ctx, q, posW, posH, opts)
	k = rope2D(ctx, k, posW, posH, opts)

	q = q.Permute(ctx, 0, 2, 1, 3)
	k = k.Permute(ctx, 0, 2, 1, 3)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := k.MulmatFullPrec(ctx, q)
	kq = kq.Scale(ctx, 1/math.Sqrt(float64(opts.headDim)))
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)

	kqv := v.Mulmat(ctx, kq)
	kqv = kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, numPatches)
	return sa.Output.Forward(ctx, kqv)
}

type VisionMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

type VisionLayer struct {
	Norm1         *nn.LayerNorm `gguf:"ln1"`
	SelfAttention *VisionAttention
	Norm2         *nn.LayerNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

func (l *VisionLayer) Forward(ctx ml.Context, hiddenState, posW, posH, mask ml.Tensor, opts *VisionOptions) ml.Tensor {
	residual := hiddenState
	hiddenState = l.Norm1.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, posW, posH, mask, opts)
	hiddenState = hiddenState.Add(ctx, residual)

	residual = hiddenState
	hiddenState = l.Norm2.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Down.Forward(ctx, l.MLP.Up.Forward(ctx, hiddenState).GELUErf(ctx))
	return hiddenState.Add(ctx, residual)
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv2D    `gguf:"patch_embd"`
	PositionEmbedding ml.Tensor     `gguf:"position_embd.weight"`
	PreLayerNorm      *nn.LayerNorm `gguf:"pre_ln"`
	Layers            []VisionLayer `gguf:"blk"`
	PostLayerNorm     *nn.LayerNorm `gguf:"post_ln"`

	*VisionOptions
}

func newVisionModel(c fs.Config) *VisionModel {
	hiddenSize := int(c.Uint("vision.embedding_length"))
	numHeads := int(c.Uint("vision.attention.head_count"))
	return &VisionModel{
		Layers: make([]VisionLayer, c.Uint("vision.block_count")),
		VisionOptions: &VisionOptions{
			hiddenSize: hiddenSize,
			numHeads:   numHeads,
			headDim:    hiddenSize / max(numHeads, 1),
			patchSize:  int(c.Uint("vision.patch_size", 14)),
			merge:      int(c.Uint("vision.spatial_merge_size", 2)),
			eps:        c.Float("vision.attention.layer_norm_epsilon", 1e-5),
			ropeBase:   10000,
		},
	}
}

// visionInputs are the host-computed indices the encoder graph needs.
type visionInputs struct {
	spPerm, invPerm, posW, posH, dsPerm []int32
	spMask                              []float32
}

// newVisionInputs groups patches into window x window blocks for the sparse
// layers, with 1-indexed RoPE positions, and gathers merge x merge neighbours
// for the pixel shuffle (llama.cpp clip.cpp, PROJECTOR_TYPE_MUSE_GLIMMER).
func newVisionInputs(gridW, gridH, window, merge int) visionInputs {
	n := gridW * gridH
	in := visionInputs{
		invPerm: make([]int32, n),
		posW:    make([]int32, n),
		posH:    make([]int32, n),
	}

	var lens []int
	for wy := 0; wy < (gridH+window-1)/window; wy++ {
		for wx := 0; wx < (gridW+window-1)/window; wx++ {
			count := 0
			for hh := range window {
				for ww := range window {
					gy, gx := wy*window+hh, wx*window+ww
					if gy < gridH && gx < gridW {
						in.spPerm = append(in.spPerm, int32(gy*gridW+gx))
						count++
					}
				}
			}
			if count > 0 {
				lens = append(lens, count)
			}
		}
	}

	for i, orig := range in.spPerm {
		in.posW[i] = orig%int32(gridW) + 1
		in.posH[i] = orig/int32(gridW) + 1
		in.invPerm[orig] = int32(i)
	}

	in.spMask = make([]float32, n*n)
	for i := range in.spMask {
		in.spMask[i] = float32(math.Inf(-1))
	}
	off := 0
	for _, s := range lens {
		for a := range s {
			for b := range s {
				in.spMask[(off+a)*n+off+b] = 0
			}
		}
		off += s
	}

	for oy := range gridH / merge {
		for ox := range gridW / merge {
			for ry := range merge {
				for rx := range merge {
					in.dsPerm = append(in.dsPerm, int32((oy*merge+ry)*gridW+ox*merge+rx))
				}
			}
		}
	}

	return in
}

// Forward encodes pixels [width, height, channels] into merged image tokens
// [hiddenSize*merge*merge, numTokens].
func (m *VisionModel) Forward(ctx ml.Context, pixels ml.Tensor, gridW, gridH int) ml.Tensor {
	n := gridW * gridH
	window := int(math.Sqrt(float64(m.PositionEmbedding.Dim(1))))
	in := newVisionInputs(gridW, gridH, window, m.merge)

	hiddenState := m.PatchEmbedding.Forward(ctx, pixels, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, n, m.hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// The learned position table covers a window x window grid; resize it to this image.
	pos := m.PositionEmbedding
	if gridW != window || gridH != window {
		pos = pos.Reshape(ctx, m.hiddenSize, window, window)
		pos = pos.Permute(ctx, 2, 0, 1, 3)
		pos = pos.InterpolateBilinear(ctx, gridW, gridH, m.hiddenSize, 1)
		pos = pos.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
		pos = pos.Reshape(ctx, m.hiddenSize, n)
	}
	hiddenState = hiddenState.Add(ctx, pos)

	hiddenState = hiddenState.Rows(ctx, ctx.Input().FromInts(in.spPerm, n))
	hiddenState = m.PreLayerNorm.Forward(ctx, hiddenState, m.eps)

	posW := ctx.Input().FromInts(in.posW, n)
	posH := ctx.Input().FromInts(in.posH, n)
	mask := ctx.Input().FromFloats(in.spMask, n, n)

	for i, layer := range m.Layers {
		var layerMask ml.Tensor
		if i != len(m.Layers)-1 && (i+1)%sparseFactor != 0 {
			layerMask = mask
		}
		hiddenState = layer.Forward(ctx, hiddenState, posW, posH, layerMask, m.VisionOptions)
	}

	hiddenState = m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = hiddenState.Rows(ctx, ctx.Input().FromInts(in.invPerm, n))

	// Pixel shuffle: each output token concatenates its merge x merge neighbours.
	numTokens := (gridW / m.merge) * (gridH / m.merge)
	hiddenState = hiddenState.Rows(ctx, ctx.Input().FromInts(in.dsPerm, n))
	hiddenState = hiddenState.Reshape(ctx, m.hiddenSize, m.merge*m.merge, numTokens)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	return hiddenState.Reshape(ctx, m.hiddenSize*m.merge*m.merge, numTokens)
}

// Projector is the adapter (mm.0, mm.1) and the language model's vision
// projection (mm.2), with an exact GELU between them.
type Projector struct {
	Linear0 *nn.Linear `gguf:"0"`
	Linear1 *nn.Linear `gguf:"1"`
	Linear2 *nn.Linear `gguf:"2"`
}

func (p *Projector) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	t = p.Linear0.Forward(ctx, t).GELUErf(ctx)
	t = p.Linear1.Forward(ctx, t).GELUErf(ctx)
	return p.Linear2.Forward(ctx, t)
}
