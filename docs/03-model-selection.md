# Why Qwen3 8B

The model choice for this 16 GB setup wasn't arbitrary. Here's the full reasoning.

---

## The constraint

A 16 GB Apple Silicon Mac has this memory budget:

| Component | Memory |
|-----------|--------|
| macOS + system overhead | ~6–8 GB |
| Hermes Agent (Python, tools, context) | ~1–2 GB |
| **Available for LLM** | **~7–9 GB** |

That ~7–9 GB has to hold both the model weights AND the KV cache (the context window's working memory). So the model itself needs to be ~5 GB or smaller at usable quantization, leaving 2–4 GB for context.

This means: **8B parameters at Q4_K_M (~5 GB) is the ceiling.** Anything bigger won't leave room for context.

---

## Why not a smaller model?

Models below 8B exist and would fit more comfortably:

| Model | Size (Q4) | Why we rejected it |
|-------|-----------|-------------------|
| Gemma 3 4B | ~2.5 GB | Limited tool calling support — trained more for chat than agent use |
| Phi-3.5 Mini 3.8B | ~2.2 GB | Good for size, but tool calling is unreliable |
| Llama 3.2 3B | ~2 GB | Too weak for multi-step agent workflows |
| Qwen3 4B | ~2.5 GB | Exists and is decent, but 8B is significantly better at reasoning |

The jump from 4B to 8B in reasoning quality is substantial. The jump from 8B to 32B is also substantial, but 32B won't fit on 16 GB. **8B is the sweet spot at this memory tier.**

---

## Why Qwen3 8B specifically

### 1. Tool calling is first-class

Qwen3 was trained with explicit tool-calling / function-calling support. The model correctly:
- Formats tool call arguments as structured JSON
- Handles multi-turn tool interactions (call → result → next call)
- Knows when to stop calling tools and give a final answer
- Handles errors gracefully (retry with different approach)

This is non-negotiable for Hermes — every action the agent takes is a tool call. A model that can't call tools reliably is useless as an agent brain, no matter how smart it is at trivia.

### 2. 128K native context support

Qwen3 8B natively supports up to 128K token context. We're using 32K (see [context length doc](04-context-length.md)), but the model architecture is designed for long context — it doesn't degrade as sharply as some models when context grows.

### 3. Strong instruction following

Qwen3 8B scores well on instruction-following benchmarks (IFEval), meaning it:
- Follows multi-step instructions without skipping steps
- Formats output as requested (markdown, code blocks, JSON)
- Doesn't hallucinate tool calls that don't exist
- Stays on task in longer conversations

### 4. Good reasoning for its size

Qwen3 8B includes thinking/reasoning capabilities. It can:
- Break down problems step by step
- Self-correct when it notices an error
- Plan multi-step tool call sequences

Is it as smart as a 70B model? No. Is it smart enough for daily agent tasks? Yes, comfortably.

### 5. Active development

Qwen is actively maintained by Alibaba's Qwen team with regular updates. The Qwen3 series was a significant jump over Qwen2.5 in reasoning and tool use.

---

## The alternatives we considered

| Model | Size (Q4) | Tool calling | Verdict |
|-------|-----------|--------------|---------|
| Llama 3.1 8B Instruct | ~5 GB | Good | Solid backup. Slightly weaker tool calling than Qwen3, slightly weaker reasoning. If Qwen3 isn't available, this is the second choice. |
| Mistral 7B Instruct v0.3 | ~4.5 GB | OK | Older model. Decent at chat, weaker at structured tool calls. Context limited to 32K. |
| Phi-4 14B | ~8 GB | Weak | Too big — barely fits, leaves almost no room for KV cache. Tool calling unreliable. |
| Gemma 3 27B | ~17 GB | Limited | Far too large for 16 GB. |
| DeepSeek-Coder 6.7B | ~4 GB | OK | Code specialist, not a general agent. Too narrow for Hermes. |

---

## Quantization choice: Q4_K_M

Q4_K_M means 4-bit quantization with medium-precision "important" tensors (the K and attention layers get slightly higher precision than the rest).

| Quantization | Size | Quality loss | Fits 16GB? |
|--------------|------|-------------|------------|
| FP16 (unquantized) | ~16 GB | None | ❌ No room for anything else |
| Q8_0 | ~8.5 GB | Minimal | ⚠️ Barely, ~1 GB for KV cache |
| **Q4_K_M** | **~5 GB** | **Small** | **✅ 3–4 GB for KV cache** |
| Q4_K_S | ~4.8 GB | Small | ✅ Slightly more KV room |
| Q3_K_M | ~3.7 GB | Noticeable | ✅ But quality drops |
| Q2_K | ~3 GB | Significant | ✅ But too degraded |

Q4_K_M is the consensus sweet spot across the local LLM community: small enough to fit comfortably, high enough quality that degradation is barely perceptible for agent tasks.

---

## What "good enough" looks like

On a 16 GB Mac with Qwen3 8B Q4_K_M, here's what you can expect:

### Works well ✅
- General chat and Q&A
- Simple tool calls: read a file, run a command, search the web
- Short code snippets and scripts
- Summarizing documents (within context limits)
- Following step-by-step instructions

### Works but with effort ⚠️
- Multi-step agent workflows (3+ sequential tool calls with reasoning between each)
- Complex code generation (works better with smaller, focused requests)
- Long conversations (context compression triggers more often)

### Struggles ❌
- Very complex reasoning chains (the model loses track of earlier constraints)
- Large codebase analysis (context fills up, quality degrades)
- Tasks requiring deep domain expertise (medical, legal, etc.)

---

## The upgrade path

If you outgrow Qwen3 8B, the path is clear:

| Your Mac | Model | Quant | Context |
|----------|-------|-------|---------|
| 16 GB | Qwen3 8B | Q4_K_M | 32K |
| 32 GB | Qwen3 32B | Q4_K_M | 128K |
| 48 GB | Qwen3 32B | Q8_0 | 64K |
| 64 GB+ | Llama 3.3 70B | Q4_K_M | 128K |

The model is the only thing that changes — Hermes, Ollama, and all your configs stay the same. Just `ollama pull qwen3:32b` and update the model name in Hermes config.