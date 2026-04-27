#!/bin/bash


GOINFRE_PATH="/home/goinfre/tmousnia"


export UV_CACHE_DIR="$HOME/goinfre/tmousnia/.cache/uv"
export HF_HOME="$HOME/goinfre/tmousnia/.cache/hugging_face"

export PATH="$HOME/.local/bin:$PATH"

echo "✅ Environment configured for machine: $(hostname)"
echo "UV Cache: $UV_CACHE_DIR"
echo "HF Home:  $HF_HOME"