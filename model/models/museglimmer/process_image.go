package museglimmer

import (
	"image"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model/imageproc"
)

type ImageProcessor struct {
	patchSize, merge int
	maxTokens        int
}

func newImageProcessor(c fs.Config) ImageProcessor {
	patchSize := int(c.Uint("vision.patch_size", 14))
	merge := int(c.Uint("vision.spatial_merge_size", 2))
	cell := patchSize * merge
	imageSize := int(c.Uint("vision.image_size", 896))
	return ImageProcessor{
		patchSize: patchSize,
		merge:     merge,
		// llama.cpp allows 4096 image tokens. A global vision layer attends
		// across every patch: 4096 tokens (16384 patches) needs ~16 GiB per
		// layer, past one K80 die, so the cap is the GGUF's native image_size.
		maxTokens: (imageSize / cell) * (imageSize / cell),
	}
}

// gridSize picks the patch grid that keeps the aspect ratio under maxTokens
// merged tokens (transformers' get_aspect_ratio_preserving_size, as llama.cpp
// ports it). It returns the target size in pixels.
func gridSize(imgW, imgH, cell, maxTokens int) (int, int) {
	nph := float64(imgH) / float64(cell)
	npw := float64(imgW) / float64(cell)
	ratio := 1.0
	if nph > 0 {
		ratio = npw / nph
	}
	if nph*npw > float64(maxTokens) {
		nph = math.Sqrt(float64(maxTokens) / ratio)
		npw = nph * ratio
	}

	hs := [2]int{int(math.Floor(nph)), int(math.Ceil(nph))}
	ws := [2]int{int(math.Floor(npw)), int(math.Ceil(npw))}
	targetAR := float64(imgH) / float64(imgW)
	bestH, bestW, bestD := -1, -1, 0.0
	for _, h := range hs {
		for _, w := range ws {
			if h < 1 || w < 1 || h*w > maxTokens {
				continue
			}
			d := math.Abs(float64(h)/float64(w) - targetAR)
			if bestH < 0 || d < bestD || (d == bestD && h*w > bestH*bestW) {
				bestH, bestW, bestD = h, w, d
			}
		}
	}
	if bestH < 0 {
		bestH = max(1, int(math.Round(nph)))
		bestW = max(1, int(math.Round(npw)))
	}
	return bestW * cell, bestH * cell
}

// ProcessImage stretches the image to the chosen grid and normalizes it to
// [-1, 1], channel-first. It returns the pixels and the patch grid.
func (p *ImageProcessor) ProcessImage(img image.Image) ([]float32, int, int) {
	cell := p.patchSize * p.merge
	w, h := gridSize(img.Bounds().Dx(), img.Bounds().Dy(), cell, p.maxTokens)
	// llama.cpp resizes with Lanczos; Catmull-Rom is the closest the fork has.
	img = imageproc.Resize(imageproc.Composite(img), image.Point{X: w, Y: h}, imageproc.ResizeCatmullrom)
	pixels := imageproc.Normalize(img, [3]float32{0.5, 0.5, 0.5}, [3]float32{0.5, 0.5, 0.5}, true, true)
	return pixels, w / p.patchSize, h / p.patchSize
}
