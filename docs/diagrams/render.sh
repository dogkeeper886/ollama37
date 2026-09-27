#!/usr/bin/env bash
# Render every SVG in this folder to png/<name>.png.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p png
for f in *.svg; do
  rsvg-convert -z 2 "$f" -o "png/${f%.svg}.png"
done
