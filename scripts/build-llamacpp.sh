#!/usr/bin/env bash
# =============================================================================
#  build-llamacpp.sh — build llama.cpp with Metal support on Apple Silicon
# =============================================================================
#  Builds llama.cpp from source, verifies the build supports quantized KV
#  caches (-ctk/-ctv), fetches Qwen3 8B Q4_K_M if it is missing, and prints the
#  64K launch command. You need a build from 2025 or later: KV cache
#  quantization is what lets 64K context fit in 16 GB.
#
#  Usage:  bash build-llamacpp.sh
# =============================================================================
set -euo pipefail

MODEL_DIR="$HOME/models"
MODEL_FILE="$MODEL_DIR/Qwen3-8B-Q4_K_M.gguf"
LLAMA_DIR="$HOME/llama.cpp"
CTX=65536

ok()   { printf '[✓] %s\n' "$*"; }
info() { printf '[…] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*"; }
die()  { printf '[✗] %s\n' "$*" >&2; exit 1; }

# Bounded "--help" probe (macOS has no GNU timeout(1)).
help_text() {
    local tmp pid i=0
    tmp="$(mktemp)"
    "$1" --help >"$tmp" 2>&1 &
    pid=$!
    while [ "$i" -lt 10 ]; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
        i=$((i + 1))
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    cat "$tmp"
    rm -f "$tmp"
}

printf '%s\n' "═══════════════════════════════════════════════════════════════"
printf '%s\n' "  llama.cpp build for Apple Silicon (Metal)"
printf '%s\n' "═══════════════════════════════════════════════════════════════"

# ---- 1. Prerequisites ------------------------------------------------------
info "Checking prerequisites..."
if ! xcode-select -p >/dev/null 2>&1; then
    die "Xcode Command Line Tools missing. Run: xcode-select --install"
fi
ok "Xcode Command Line Tools present"

if ! command -v cmake >/dev/null 2>&1; then
    command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install from https://brew.sh first."
    info "Installing CMake via Homebrew..."
    brew install cmake
fi
ok "CMake: $(cmake --version | head -1)"

# ---- 2. Source -------------------------------------------------------------
if [ -d "$LLAMA_DIR/.git" ]; then
    info "Updating existing llama.cpp checkout..."
    git -C "$LLAMA_DIR" pull --ff-only || warn "git pull failed — building what is on disk"
else
    info "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
ok "Source ready at $LLAMA_DIR (commit $(git -C "$LLAMA_DIR" rev-parse --short HEAD))"

# ---- 3. Build with Metal ---------------------------------------------------
info "Building with Metal support (2–5 minutes)..."
mkdir -p "$LLAMA_DIR/build"
cd "$LLAMA_DIR/build"
CPU_CORES="$(sysctl -n hw.ncpu)"
info "Using $CPU_CORES CPU cores"

cmake "$LLAMA_DIR" \
    -DGGML_METAL=ON \
    -DGGML_METAL_USE_BMETAL=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j "$CPU_CORES"
ok "Build complete"

# ---- 4. Verify the build can do what we need -------------------------------
SERVER="$LLAMA_DIR/build/bin/llama-server"
[ -x "$SERVER" ] || die "llama-server not found at $SERVER — build failed."

SERVER_HELP="$(help_text "$SERVER")"
case "$SERVER_HELP" in
    *--cache-type-k*) ok "Build supports quantized KV cache (-ctk/-ctv)" ;;
    *) die "This build has no --cache-type-k option, so it cannot hold 64K in 16 GB.
    Make sure you cloned the current master branch and rebuilt from scratch:
        rm -rf $LLAMA_DIR/build && bash $0" ;;
esac

case "$SERVER_HELP" in
    *--flash-attn*) ok "Build supports --flash-attn (required for a quantized V cache)" ;;
    *) warn "No --flash-attn flag found — an old build. A quantized V cache needs it." ;;
esac

# ---- 5. Model --------------------------------------------------------------
if [ -f "$MODEL_FILE" ]; then
    ok "Model present: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
else
    warn "Model missing at $MODEL_FILE"
    mkdir -p "$MODEL_DIR"
    if command -v hf >/dev/null 2>&1; then
        info "Downloading Qwen3 8B Q4_K_M (~5 GB) with the Hugging Face CLI..."
        hf download Qwen/Qwen3-8B-GGUF --include '*Q4_K_M*' --local-dir "$MODEL_DIR"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        info "Downloading Qwen3 8B Q4_K_M (~5 GB)..."
        huggingface-cli download Qwen/Qwen3-8B-GGUF --include '*Q4_K_M*' --local-dir "$MODEL_DIR"
    else
        cat <<EOF

  No Hugging Face CLI found. Either install it:
      pipx install 'huggingface_hub[cli]'
  or download the Q4_K_M GGUF by hand from:
      https://huggingface.co/Qwen/Qwen3-8B-GGUF
  and save it as: $MODEL_FILE
EOF
    fi
fi

# ---- 6. Hand off -----------------------------------------------------------
cat <<EOF

═══════════════════════════════════════════════════════════════
  ✅ Build finished — start it at 64K with the launcher
═══════════════════════════════════════════════════════════════

  bash scripts/launch-llm-64k.sh

It runs this, checks the server is healthy, and confirms it really reports
${CTX} tokens before telling you the Hermes settings:

  $SERVER \\
    -m $MODEL_FILE \\
    -c $CTX -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 \\
    --host 127.0.0.1 --port 8080

  Then, in Hermes: custom endpoint, base_url http://127.0.0.1:8080/v1,
  model qwen3-8b, context_length ${CTX}.

  Full walkthrough: docs/02-llamacpp-setup.md
═══════════════════════════════════════════════════════════════
EOF
