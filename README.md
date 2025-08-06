# RAG

GPT‑Small & Zero‑Shot RAG Toolkit

이 저장소는 직접 구현한 GPT‑small 언어 모델과 그 위에 얹힌 Zero‑Shot Retrieval‑Augmented Generation (RAG) 실험을 아카이빙합니다. 아래 설명은 RAG의 동작 원리, 성능 개선 실험, 그리고 재현 방법을 정리해 둔 것입니다.

1. GPT‑small

구조 : 12 layers · 12 heads · 768 hidden (≈124 M 파라미터)

코드 : gpt_small/ – 순수 PyTorch, rotary positional embedding, FP16 지원

2. RAG란?

Retrieval‑Augmented Generation은 ▶ ① 검색기(Retriever)가 질의와 관련된 문서를 가져오고 ▶ ② 생성기(Generator)가 "질문 + 문서"를 입력으로 받아 답변을 생성하는 두 단계 파이프라인입니다. 따라서 언어 모델이 학습하지 못한 지식도 실시간으로 반영할 수 있습니다.

파이프라인 구성

Stage

구현체

주요 설정

Retriever

Pyserini BM25 (wikipedia‑dpr‑100w)

top‑k=5, 한국어·영어 혼합 검색 지원

Fusion

FiD‑style passage concatenation

passage 사이에 <sep> 토큰 삽입

Generator

GPT‑small (Instruction Tuned)

gradient checkpointing, max input = 2048 tokens

3. Zero‑Shot 성능 향상 실험

3‑1. Prompt Engineering

Variant

핵심 아이디어

예시 (질문: "Who founded Victoria's Secret?")

Baseline

단순 Q → A 형태

Who founded Victoria's Secret?

User/Assistant

Llama 3.2 대화 포맷 채택

```<

start_header_id

>user<

end_header_id

>

Who founded Victoria's Secret?

<

start_header_id

>assistant<

end_header_id

>```

Chain‑of‑Thought (CoT)

추론 과정을 강제 (step‑by‑step)

위 포맷 + "Let's think step by step."

3‑2. Retrieval Tuning

top‑k를 3→5로 확대하면서도 중복 passage 제거 로 잡음 감소

질문 길이에 따라 idf 가중치를 조정해 긴 질문 penalty 완화

3‑3. Answer Post‑processing

소문자화, 구두점 제거, 숫자 normalize 로 Exact Match(EM) 계산 일관성 확보

3‑4. 결과 (Natural Questions dev)

Prompt

EM

Accuracy

Baseline

14.2

0.09

User/Assistant

18.7

0.17

CoT (step‑by‑step)

22.1

0.19

User/Assistant + CoT (최종)

23.5

0.20
