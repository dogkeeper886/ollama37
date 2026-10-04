package model

import (
	"reflect"
	"testing"
)

func TestParseGGUFGenerationDefaults(t *testing.T) {
	kv := map[string]float64{
		"general.sampling.top_k": 20,
		"general.sampling.top_p": 0.95,
		"general.sampling.temp":  0,
	}
	lookup := func(key string) (float64, bool) { v, ok := kv[key]; return v, ok }

	got := ParseGGUFGenerationDefaults(
		func(key string) (int64, bool) { v, ok := lookup(key); return int64(v), ok },
		lookup,
	)
	want := GenerationDefaults{"top_k": int64(20), "top_p": 0.95, "temperature": 0.0}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestParseGGUFGenerationDefaultsNone(t *testing.T) {
	none := func(string) (float64, bool) { return 0, false }
	got := ParseGGUFGenerationDefaults(func(string) (int64, bool) { return 0, false }, none)
	if got != nil {
		t.Errorf("got %#v, want nil", got)
	}
}
