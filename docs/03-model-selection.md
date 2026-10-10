# Why Qwen3 8B

The model choice was not arbitrary. It came out of the memory budget left after
satisfying Hermes' 64K context requirement.

---

## The constraint, in one budget

A 16 GB Apple Silicon Mac has to hold, at the same time:

| Component | Size |
|-----------|------|
| Model weights | as small as you like |
| KV cache for 64K tokens | **fixed by the model's architecture** |
| Metal compute buffers | ~1 GiB |
| macOS + Hermes + apps | 4–6 GiB |

Weights are negotiable through quantization. The KV cache at 64K is 4.78 GiB at
q8_0 for any Qwen3 model with 36 layers, 8 KV heads and head_dim 128 — *including
the small ones*. That asymmetry drives the whole decision.

---

## Why not a smaller model?

| Model | Weights (Q4) | KV @ 64K, q8_0 | Why we did not pick it |
|-------|--------------|----------------|------------------------|
| Qwen3 4B | ~2.5 GB | 4.78 GiB | Same KV shape as 8B — saves weights only, and is noticeably weaker at tool use |
| Gemma 3 4B | ~2.5 GB | varies | Weaker agent/tool-calling behavior |
| Phi-4 mini 3.8B | ~2.2 GB | varies | Tool calling unreliable for agent loops |
| Llama 3.2 3B | ~2 GB | varies | Too weak for multi-step workflows |

Dropping from 8B to 4B buys ~2.2 GB and costs real reasoning quality, while the
64K cache costs the same either way. If you need memory back, quantize the cache
(q4_0 → 2.53 GiB) instead: it costs long-context precision, not general
intelligence.

---

## Why Qwen3 8B specifically

**1. First-class tool calling.** Qwen3 was trained for function calling: it emits
well-formed JSON arguments, chains call → result → call, decides when to stop, and
recovers from errors. Every Hermes action is a tool call, so this is the one
requirement that cannot be traded away.

**2. Long-context behavior.** The GGUF declares a 40,960-token trained window and
the model extends past it with RoPE/YaRN scaling, so 64K is reachable without
quality collapse (the launcher applies the scaling). Models with short windows
cannot be stretched this far.

**3. Instruction following.** Strong IFEval-style behavior: follows multi-step
instructions, respects output formats, and does not hallucinate tools.

**4. Reasoning for its size.** Thinks step by step, self-corrects, plans tool
sequences. Not a 70B — enough for daily agent work.

**5. Ecosystem.** Converted GGUFs, active maintenance, a current chat template
with tool support built in.

---

## Why Q4_K_M

| Quant | Weights | Quality | Fits 16 GB at 64K? |
|-------|---------|---------|--------------------|
| FP16 | ~16 GB | reference | No |
| Q8_0 | ~8.5 GB | near-lossless | 8.5 + 4.78 cache + 1 = 14.3 GiB — no room for macOS |
| **Q4_K_M** | **~4.7 GiB** | **small loss** | **4.7 + 4.78 + 1 = 10.5 GiB — yes** |
| Q4_K_S | ~4.5 GiB | small loss | Yes, marginal gain |
| Q3_K_M | ~3.7 GiB | noticeable | Yes, quality drops |
| Q2_K | ~3 GiB | significant | Yes, not worth it |

The arithmetic is unforgiving: at 64K the cache consumes 4.78 GiB before weights
are counted, so an 8.5 GB Q8_0 model plus macOS does not fit. Q4_K_M is the only
quantization that leaves a workable margin — and at the 4-bit level the quality
difference is far smaller than the difference between running 64K and not running
Hermes at all.

---

## The alternatives we considered

| Model | Weights (Q4) | Tool calling | Verdict |
|-------|--------------|--------------|---------|
| Llama 3.1 8B Instruct | ~5 GB | Good | Solid fallback; slightly weaker tool use and reasoning |
| Mistral 7B Instruct v0.3 | ~4.5 GB | OK | Older, weaker at structured calls |
| Phi-4 14B | ~8 GB | Weak | Too big once the 4.78 GiB cache is counted |
| Qwen3 14B | ~9 GB | Good | Best 16 GB upgrade if you drop the cache to q4_0 and close everything |
| Llama 3.3 70B | ~40 GB | Strong | Needs 64 GB+ |

---

## What "good enough" looks like

**Works well** ✅
- Chat, Q&A, summarization
- Tool calls: read a file, run a command, search the web
- Multi-step agent loops with a genuinely large context window
- Focused code generation

**Works with effort** ⚠️
- Long reasoning chains (an 8B ceiling, not a context ceiling)
- Whole-codebase analysis
- Very long documents — the window holds them, attention quality still fades at the extremes

**Struggles** ❌
- Deep domain reasoning where a 70B is simply better
- Heavy parallel workloads alongside Hermes on 16 GB

---

## The upgrade path

| Your Mac | Model | Quant | KV cache | Context |
|----------|-------|-------|----------|---------|
| 16 GB | Qwen3 8B | Q4_K_M | q8_0 | 64K |
| 24 GB | Qwen3 14B | Q4_K_M | q8_0 | 64K+ |
| 48 GB | Qwen3 14B | Q8_0 | q8_0 | 128K |
| 64 GB+ | Llama 3.3 70B | Q4_K_M | f16 | 128K |

Swap the model file, keep the flags. Hermes and the configs do not change.
