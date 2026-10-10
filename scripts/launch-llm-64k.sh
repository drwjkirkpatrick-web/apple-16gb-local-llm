#!/usr/bin/env bash
# =============================================================================
#  launch-llm-64k.sh — llama-server at 64K context with an 8-bit KV cache
# =============================================================================
#  Why this exists:
#    Hermes Agent refuses to start against a model with less than 64,000 tokens
#    of context. That is a hard constant, not a preference:
#
#        # agent/model_metadata.py
#        MINIMUM_CONTEXT_LENGTH = 64_000
#
#    So "good enough" starts at 64K. This script starts Qwen3 8B at 65,536
#    tokens of context with a q8_0 KV cache so the whole thing still fits on a
#    16 GB Apple Silicon Mac.
#
#  Memory at 64K, Qwen3 8B (36 layers, 8 KV heads, head_dim 128):
#        model weights (Q4_K_M)         ~5.0 GB
#        KV cache, q8_0 @ 65,536 tok    4.78 GiB   (76.5 KiB per token)
#        KV cache, f16  @ 65,536 tok    9.00 GiB   (too big to leave room)
#        compute buffers                ~1 GB
#        ------------------------------------------
#        llama-server total             ~10.5 GiB  of 16 GB
#
#  Usage:
#        bash launch-llm-64k.sh                     # defaults below
#        PORT=8090 bash launch-llm-64k.sh           # different port
#        MODEL=~/models/other.gguf bash launch-llm-64k.sh
#        CACHE_K=q4_0 CACHE_V=q4_0 bash launch-llm-64k.sh   # last-resort squeeze
#
#  Overridable environment variables:
#        MODEL  LLAMA_SERVER  PORT  HOST  NGL  CTX
#        CACHE_K  CACHE_V  PARALLEL  YARN (auto|on|off)  JINJA (1|0)  LOG
# =============================================================================
set -euo pipefail

CTX="${CTX:-65536}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
NGL="${NGL:-99}"
CACHE_K="${CACHE_K:-q8_0}"
CACHE_V="${CACHE_V:-q8_0}"
PARALLEL="${PARALLEL:-1}"       # each slot gets its own KV cache — keep this at 1
YARN="${YARN:-auto}"            # auto = only scale RoPE if CTX exceeds the trained window
JINJA="${JINJA:-1}"             # use the GGUF's own chat template (Qwen3 tool calls)
LOG="${LOG:-/tmp/llama-server-${PORT}.log}"
SERVER="${LLAMA_SERVER:-}"

HERMES_MIN_CTX=64000

# Model candidates, in order. First one that exists wins.
MODEL_CANDIDATES=(
    "${MODEL:-}"
    "$HOME/models/Qwen3-8B-Q4_K_M.gguf"
    "$HOME/models/qwen3-8b-q4_k_m.gguf"
    "$HOME/models/Qwen3-8B-Q4_K_M/Qwen3-8B-Q4_K_M.gguf"
)

say()  { printf '%s\n' "$*"; }
ok()   { printf '[✓] %s\n' "$*"; }
info() { printf '[…] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*"; }
die()  { printf '[✗] %s\n' "$*" >&2; exit 1; }

# Bounded, portable "--help" probe. macOS has no GNU timeout(1), and blindly
# piping a server binary into grep can hang if it does not exit on --help.
help_text() {
    local tmp pid
    tmp="$(mktemp)"
    "$SERVER" --help >"$tmp" 2>&1 &
    pid=$!
    local i=0
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

say "═══════════════════════════════════════════════════════════════"
say "  llama.cpp @ ${CTX} context, ${CACHE_K}/${CACHE_V} KV cache"
say "═══════════════════════════════════════════════════════════════"

# ---- 0. Refuse to launch something Hermes will reject -----------------------
if [ "$CTX" -lt "$HERMES_MIN_CTX" ]; then
    warn "CTX=${CTX} is below the Hermes floor of ${HERMES_MIN_CTX} tokens."
    warn "Hermes rejects the model at startup (MINIMUM_CONTEXT_LENGTH = 64000)."
    die  "Raise CTX (e.g. CTX=65536) — do not lower it."
fi
ok "Context ${CTX} ≥ Hermes floor ${HERMES_MIN_CTX}"

# ---- 1. Find llama-server --------------------------------------------------
if [ -z "$SERVER" ]; then
    for candidate in \
        "$HOME/llama.cpp/build/bin/llama-server" \
        "/opt/homebrew/bin/llama-server" \
        "/usr/local/bin/llama-server"
    do
        if [ -x "$candidate" ]; then SERVER="$candidate"; break; fi
    done
fi
[ -n "$SERVER" ] || SERVER="$(command -v llama-server 2>/dev/null || true)"
[ -n "$SERVER" ] && [ -x "$SERVER" ] || die "llama-server not found. Build it with scripts/build-llamacpp.sh, or set LLAMA_SERVER=/path/to/llama-server"
ok "llama-server: $SERVER"

# KV cache quantization needs a 2025-or-later build. Check before we start.
SERVER_HELP="$(help_text)"
case "$SERVER_HELP" in
    *--cache-type-k*)
        ok "Build supports quantized KV cache (-ctk/-ctv)" ;;
    *)
        if [ -z "$SERVER_HELP" ]; then
            warn "Could not read --help output from $SERVER — skipping the build check."
        else
            die "$SERVER does not support --cache-type-k (KV cache quantization).
    Rebuild from source: scripts/build-llamacpp.sh
    Old builds cannot hold a 64K q8_0 cache on 16 GB."
        fi ;;
esac

# ---- 2. Find the model -----------------------------------------------------
MODEL_FILE=""
for candidate in "${MODEL_CANDIDATES[@]}"; do
    [ -n "$candidate" ] || continue
    if [ -f "$candidate" ]; then MODEL_FILE="$candidate"; break; fi
done

if [ -z "$MODEL_FILE" ]; then
    warn "No model file found. Looked for:"
    for candidate in "${MODEL_CANDIDATES[@]}"; do [ -n "$candidate" ] && say "      $candidate"; done
    say ""
    say "  Download Qwen3 8B Q4_K_M (~5 GB):"
    say "      mkdir -p ~/models"
    say "      hf download Qwen/Qwen3-8B-GGUF --include '*Q4_K_M*' --local-dir ~/models"
    say ""
    say "  Then re-run this script (or pass MODEL=/path/to/file.gguf)."
    exit 1
fi
ok "Model: $MODEL_FILE"

# ---- 3. Read the model's real shape from its GGUF header -------------------
read -r GEN_LAYERS GEN_KVHEADS GEN_HEADDIM GEN_NATIVE <<<"$(
python3 - "$MODEL_FILE" <<'PY' 2>/dev/null || echo "36 8 128 40960"
import struct, sys

path = sys.argv[1]
VT = {0:("B",1),1:("b",1),2:("H",2),3:("h",2),4:("I",4),5:("i",4),
      6:("f",4),7:("?",1),10:("Q",8),11:("q",8),12:("d",8)}

with open(path, "rb") as f:
    head = f.read(24)
    if head[:4] != b"GGUF":
        sys.exit(2)
    _ver, = struct.unpack_from("<I", head, 4)
    _n_tensors, = struct.unpack_from("<Q", head, 8)
    n_kv, = struct.unpack_from("<Q", head, 16)

    def read(fmt):
        size = struct.calcsize(fmt)
        buf = f.read(size)
        if len(buf) < size:
            raise EOFError
        return struct.unpack(fmt, buf)

    def read_str():
        (n,) = read("<Q")
        return f.read(n).decode("utf-8", "replace")

    def read_val(t):
        if t == 8:
            return read_str()
        if t == 9:
            (et,) = read("<I")
            (cnt,) = read("<Q")
            return [read_val(et) for _ in range(min(cnt, 4096))]
        fmt, _ = VT[t]
        return read("<" + fmt)[0]

    meta = {}
    try:
        for _ in range(n_kv):
            key = read_str()
            (t,) = read("<I")
            meta[key] = read_val(t)
    except (EOFError, struct.error, KeyError):
        pass

arch = meta.get("general.architecture", "qwen3")
def g(name, default):
    return meta.get(f"{arch}.{name}", default)

layers = int(g("block_count", 36) or 36)
kv_heads = int(g("attention.head_count_kv", 8) or 8)
heads = int(g("attention.head_count", 32) or 32)
embed = int(g("embedding_length", 4096) or 4096)
head_dim = int(g("attention.key_length", embed // heads) or (embed // heads))
native = int(g("context_length", 40960) or 40960)

print(layers, kv_heads, head_dim, native)
PY
)"

GEN_LAYERS="${GEN_LAYERS:-36}"; GEN_KVHEADS="${GEN_KVHEADS:-8}"
GEN_HEADDIM="${GEN_HEADDIM:-128}"; GEN_NATIVE="${GEN_NATIVE:-40960}"

WEIGHTS_BYTES="$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE")"

# ---- 4. Memory budget, computed from the real file + real shape ------------
BUDGET="$(python3 - "$WEIGHTS_BYTES" "$CTX" "$GEN_LAYERS" "$GEN_KVHEADS" "$GEN_HEADDIM" "$CACHE_K" "$CACHE_V" <<'PY'
import sys
weights, ctx, layers, kv_heads, head_dim, ck, cv = sys.argv[1:8]
weights, ctx, layers, kv_heads, head_dim = (int(weights), int(ctx), int(layers),
                                            int(kv_heads), int(head_dim))

# bytes per element, including block overhead
# Bytes per stored element, including each block's scale/min overhead.
# q8_0  = 2 byte scale + 32 x 8-bit   per 32 elements -> 34/32
# q4_0  = 2 byte scale + 32 x 4-bit   per 32 elements -> 18/32
# q4_1  = 2 scale + 2 min + 32 x 4bit per 32 elements -> 20/32
PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34/32,
            "q4_0": 18/32, "q4_1": 20/32, "iq4_nl": 18/32,
            "q5_0": 22/32, "q5_1": 24/32}

def side_bytes(kind):
    """Bytes for ONE of K or V: layers * kv_heads * head_dim * ctx * bytes/elem."""
    return layers * kv_heads * head_dim * ctx * PER_ELEM.get(kind, 2.0)

kv = side_bytes(ck) + side_bytes(cv)                    # K and V can differ in type
per_token_kib = kv / ctx / 1024
compute = 1.0 * 1024**3                                 # ~1 GB Metal scratch/KQ buffers
total = weights + kv + compute
GIB = 1024**3
print(f"{weights/GIB:.2f} {kv/GIB:.2f} {total/GIB:.2f} {16 - total/GIB:.2f} {per_token_kib:.1f}")
PY
)"

read -r W_GIB KV_GIB TOT_GIB FREE_GIB KIB_PER_TOKEN <<<"$BUDGET"

say ""
say "  Memory budget for ${CTX} tokens on this model"
say "  ──────────────────────────────────────────────"
say "    weights (file on disk)        ${W_GIB} GiB"
say "    KV cache ${CACHE_K}/${CACHE_V} @ ${CTX}    ${KV_GIB} GiB   (${KIB_PER_TOKEN} KiB/token)"
say "    compute buffers               ~1.00 GiB"
say "    ────────────────────────────────────────────"
say "    llama-server total            ${TOT_GIB} GiB"
say "    left for macOS + Hermes       ${FREE_GIB} GiB of 16 GB"
say ""

FREE_INT="${FREE_GIB%.*}"
if [ "${FREE_INT:-0}" -lt 2 ]; then
    warn "Almost nothing left for macOS — expect sustained swapping."
    warn "Quit everything else, or switch to CACHE_K=q4_0 CACHE_V=q4_0 (≈2.5 GiB at ${CTX})."
elif [ "${FREE_INT:-0}" -lt 4 ]; then
    warn "Tight: ${FREE_GIB} GiB left for macOS + Hermes. Quit browsers, Docker and"
    warn "Electron apps before long sessions, or use CACHE_K=q4_0 CACHE_V=q4_0."
else
    ok "Workable: ${FREE_GIB} GiB stays free for macOS + Hermes on a 16 GB Mac."
    info "That is the real cost of Hermes' 64K floor — keep browsers closed."
fi

# ---- 5. Assemble the command ----------------------------------------------
ARGS=(
    -m "$MODEL_FILE"
    -c "$CTX"
    -ngl "$NGL"
    -fa on
    -ctk "$CACHE_K"
    -ctv "$CACHE_V"
    -np "$PARALLEL"
    --host "$HOST"
    --port "$PORT"
)
[ "$JINJA" = "1" ] && ARGS+=(--jinja)

# RoPE scaling: Qwen3 8B's GGUF declares a 40,960-token trained window. Asking
# for 65,536 without YaRN gives degraded attention past that mark.
NEEDS_YARN=0
[ "$CTX" -gt "$GEN_NATIVE" ] && NEEDS_YARN=1
if [ "$YARN" = "on" ] || { [ "$YARN" = "auto" ] && [ "$NEEDS_YARN" = "1" ]; }; then
    SCALE="$(python3 -c "print(f'{$CTX/$GEN_NATIVE:.4f}')")"
    ARGS+=(--rope-scaling yarn --rope-scale "$SCALE" --yarn-orig-ctx "$GEN_NATIVE")
    info "RoPE: YaRN scaling ${SCALE}x from a ${GEN_NATIVE}-token trained window"
    info "      (set YARN=off if you know this GGUF already bakes in its scaling)"
elif [ "$NEEDS_YARN" = "1" ]; then
    warn "CTX ${CTX} exceeds the ${GEN_NATIVE}-token trained window and YaRN is off."
fi

say ""
say "  Launching:"
say "    $SERVER ${ARGS[*]}"
say "    log: $LOG"
say ""

# ---- 6. Start it and wait for health --------------------------------------
"$SERVER" "${ARGS[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

READY=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 1
done

if [ "$READY" != "1" ]; then
    warn "Server did not become healthy. Last lines of $LOG:"
    tail -n 25 "$LOG" >&2 || true
    exit 1
fi
ok "Server healthy on http://${HOST}:${PORT}/v1"

# ---- 7. Verify the served context — the whole point of this script --------
SERVED_CTX="$(curl -sf "http://127.0.0.1:${PORT}/props" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
candidates = [
    d.get("n_ctx"),
    (d.get("default_generation_settings") or {}).get("n_ctx"),
    (d.get("default_generation_settings") or {}).get("params", {}).get("n_ctx"),
]
for c in candidates:
    if isinstance(c, int) and c > 0:
        print(c); break
' 2>/dev/null || true)"

if [ -n "$SERVED_CTX" ]; then
    if [ "$SERVED_CTX" -ge "$HERMES_MIN_CTX" ]; then
        ok "Server reports n_ctx = ${SERVED_CTX} — above the Hermes floor"
    else
        die "Server reports n_ctx = ${SERVED_CTX}, below the Hermes floor of ${HERMES_MIN_CTX}.
    Hermes will refuse to start. Check the -c value and the log: $LOG"
    fi
else
    warn "Could not read n_ctx from /props on this build — verify in the log:"
    warn "    grep -i 'n_ctx' $LOG"
fi

RSS_MB="$(ps -o rss= -p "$SERVER_PID" 2>/dev/null | awk '{printf "%d", $1/1024}')"
[ -n "$RSS_MB" ] && ok "Resident memory now: ${RSS_MB} MB"

# ---- 8. Tell the user how to wire it into Hermes --------------------------
say ""
say "═══════════════════════════════════════════════════════════════"
say "  ✅ Ready — point Hermes at it"
say "═══════════════════════════════════════════════════════════════"
say ""
say "  hermes model   → Custom endpoint (self-hosted / vLLM / llama.cpp)"
say "      Base URL:       http://127.0.0.1:${PORT}/v1"
say "      API key:        (leave blank)"
say "      Model:          qwen3-8b"
say "      Context length: ${CTX}     ← must be ≥ 64000"
say ""
say "  Or in ~/.hermes/config.yaml:"
say ""
say "      model:"
say "        provider: custom"
say "        base_url: \"http://127.0.0.1:${PORT}/v1\""
say "        api_key: \"\""
say "        default: \"qwen3-8b\""
say "        context_length: ${CTX}"
say ""
say "  Test:  hermes chat -q 'What tools do you have available?'"
say ""
say "  Ctrl+C to stop the server."
say "═══════════════════════════════════════════════════════════════"

wait "$SERVER_PID"
