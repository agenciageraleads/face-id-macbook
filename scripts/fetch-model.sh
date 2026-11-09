#!/usr/bin/env bash
# Build the CoreML models from Apache-2.0 sources: SFace (recognition) and
# Silent-Face-Anti-Spoofing (photo and screen detection).
#
# Usage: scripts/fetch-model.sh [--force]
#
# Needs Python 3.12+ (Homebrew python3.12 works). Everything else is installed
# into Tools/Conversion/.venv.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONVERSION="$ROOT/Tools/Conversion"
VENV="$CONVERSION/.venv"
MODELS="$CONVERSION/models"
ONNX="$MODELS/face_recognition_sface_2021dec.onnx"
OUT="$ROOT/Resources/FaceEmbedding.mlpackage"
COMPILED="$ROOT/Resources/FaceEmbedding.mlmodelc"
ONNX_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

FAS_COMMIT="b6d5f04ad78778917853b25c778acef6d5626d15"
FAS_BASE="https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/$FAS_COMMIT"
FAS_DIR="$MODELS/silent-face"

fetch_verified() { # url destination sha256
  if [[ ! -f "$2" ]]; then
    curl -fL --progress-bar -o "$2" "$1"
  fi
  echo "$3  $2" | shasum -a 256 -c - >/dev/null || { rm -f "$2"; echo "error: checksum mismatch for $2" >&2; exit 1; }
}

if [[ -d "$COMPILED" && -d "$ROOT/Resources/FASV2.mlmodelc" && -d "$ROOT/Resources/FASV1SE.mlmodelc" && "$FORCE" -eq 0 ]]; then
  echo "CoreML models already exist (use --force to rebuild)"
  exit 0
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "error: need Python 3.12+ (brew install python@3.12)" >&2
  exit 1
fi
echo "using $($PYTHON --version)"

if [[ ! -d "$VENV" ]]; then
  echo "creating venv..."
  "$PYTHON" -m venv "$VENV"
fi

echo "installing conversion dependencies (first run downloads PyTorch)..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet coremltools onnx onnxruntime onnx2torch torch numpy pillow

mkdir -p "$MODELS"
if [[ ! -f "$ONNX" ]]; then
  echo "downloading SFace ONNX (37 MB)..."
  curl -fL --progress-bar -o "$ONNX" "$ONNX_URL"
fi

echo "converting to CoreML..."
"$VENV/bin/python" "$CONVERSION/convert_sface.py" --onnx "$ONNX" --out "$OUT"

echo "downloading Silent-Face-Anti-Spoofing (pinned, checksum verified)..."
mkdir -p "$FAS_DIR"
fetch_verified "$FAS_BASE/src/model_lib/MiniFASNet.py" "$FAS_DIR/MiniFASNet.py" e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7
fetch_verified "$FAS_BASE/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth" "$FAS_DIR/2.7_80x80_MiniFASNetV2.pth" a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0
fetch_verified "$FAS_BASE/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth" "$FAS_DIR/4_0_0_80x80_MiniFASNetV1SE.pth" 84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990

echo "converting anti-spoofing models to CoreML..."
"$VENV/bin/python" "$CONVERSION/convert_antispoof.py" --source "$FAS_DIR" --out "$ROOT/Resources"

echo "done: $COMPILED"
echo "now run: xcodegen generate"
