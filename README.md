# GPT‑Small & Zero‑Shot RAG

This repository archives two main experiments.

1. GPT‑Small

Code: gpt_small/ – a lightweight language model implemented in pure PyTorch

2. Retrieval‑Augmented Generation (RAG)

Concept: A retriever fetches passages related to a query, and a generator answers by conditioning on the concatenation of question + passages.

Setup: BM25 retrieval (Pyserini) + GPT‑Small decoder

3. Zero‑Shot Performance Boost

User/Assistant prompt: Adopted Llama‑style headers <|start_header_id|>user<|end_header_id|> / <|start_header_id|>assistant<|end_header_id|>

Chain‑of‑Thought (CoT): Added the instruction “Let’s think step by step.” to elicit reasoning traces

This repository is for personal research and archival purposes.
