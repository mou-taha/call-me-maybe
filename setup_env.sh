#!/bin/bash


GOINFRE_PATH="/home/goinfre/tmousnia"

mkdir -p "$GOINFRE_PATH/.cache/uv"
mkdir -p "$GOINFRE_PATH/.cache/hugging_face"

export UV_CACHE_DIR="$GOINFRE_PATH/.cache/uv"
export HF_HOME="$GOINFRE_PATH/.cache/hugging_face"

export PATH="$HOME/.local/bin:$PATH"

echo "✅ Environment configured for machine: $(hostname)"
echo "UV Cache: $UV_CACHE_DIR"
echo "HF Home:  $HF_HOME"