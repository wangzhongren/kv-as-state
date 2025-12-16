# KV Cache as Inference State

This repo demonstrates that KV cache in transformer-based LLMs is a replayable inference state,
not merely a transient acceleration cache.

By serializing and restoring `past_key_values`, the model continues coherent generation
without replaying previous tokens.

This shows that conversational memory is encoded in inference state rather than text history.

File: `personal_memory_replay.py`

This is a minimal, conceptual experiment — not a production memory system.
