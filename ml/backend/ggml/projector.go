package ggml

import (
	"cmp"
	"fmt"
	"os"
	"strings"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// tensorSource locates a tensor's data: the file it lives in and the offset
// of that file's tensor data section.
type tensorSource struct {
	path       string
	dataOffset uint64
}

// qwen3vlMergerRenames maps llama.cpp mmproj (clip) tensor names to the names
// the Go qwen3vl vision model loads. It is the inverse of the renames Ollama's
// llama-server compat layer applies to Ollama-format qwen35 vision towers.
// The fused attn_qkv and the two patch_embd slices keep their clip names; the
// vision model reads those directly.
//
// gemma4v needs no table: its v.* and mm.* names already match what the Go
// gemma4 vision model loads. gemma4uv is not accepted — it carries a patch
// embedder rather than a vision tower, which that model has no path for.
var qwen3vlMergerRenames = []struct{ from, to string }{
	{"v.position_embd.", "v.pos_embed."},
	{"v.post_ln.", "v.merger.norm."},
	{"mm.0.", "v.merger.linear_fc1."},
	{"mm.2.", "v.merger.linear_fc2."},
	{".ffn_up.", ".mlp.linear_fc1."},
	{".ffn_down.", ".mlp.linear_fc2."},
	{".ln1.", ".norm1."},
	{".ln2.", ".norm2."},
	{"v.patch_embd.weight.1", "v.patch_embd_1.weight"},
}

// loadProjector decodes a separate vision projector GGUF and folds it into
// the model's metadata: clip.vision.* keys become <arch>.vision.* (without
// overriding keys the model already has) and its tensors are returned renamed
// for the model, each paired with its location in the projector file.
func loadProjector(path string, kv fsggml.KV) ([]*fsggml.Tensor, tensorSource, error) {
	r, err := os.Open(path)
	if err != nil {
		return nil, tensorSource{}, err
	}
	defer r.Close()

	proj, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, tensorSource{}, err
	}

	if arch := proj.KV().Architecture(); arch != "clip" {
		return nil, tensorSource{}, fmt.Errorf("unsupported projector architecture %q", arch)
	}

	// qwen3vl_merger declares its type at clip.projector_type, gemma4v at
	// clip.vision.projector_type. fsggml prefixes the architecture, which is
	// clip either way.
	projectorType := cmp.Or(proj.KV().String("projector_type"), proj.KV().String("vision.projector_type"))
	var renames []struct{ from, to string }
	switch projectorType {
	case "qwen3vl_merger":
		renames = qwen3vlMergerRenames
	case "gemma4v":
	default:
		return nil, tensorSource{}, fmt.Errorf("unsupported projector type %q", projectorType)
	}

	prefix := kv.Architecture() + ".vision."
	for k, v := range proj.KV() {
		if name, ok := strings.CutPrefix(k, "clip.vision."); ok {
			if _, exists := kv[prefix+name]; !exists {
				kv[prefix+name] = v
			}
		}
	}

	tensors := proj.Tensors().Items()
	for _, t := range tensors {
		for _, r := range renames {
			t.Name = strings.Replace(t.Name, r.from, r.to, 1)
		}
	}

	return tensors, tensorSource{path: path, dataOffset: proj.Tensors().Offset}, nil
}
