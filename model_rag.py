from utils.etc import hit2docdict
import torch
# Modify

class ModelRAG():
    def __init__(self):
        pass
    # 사용할 LM 선언    
    def set_model(self, model):
        self.model = model
    # 검색기
    def set_retriever(self, retriever):
        self.retriever = retriever
    # 토크나이저 선언
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    # 문서 검색
    def search(self, queries, qids, k=5):
        hits_dict = self.retriever.batch_search(queries, qids, k, threads=min(8, len(queries)))

        passages_all = []
        scores_all = []

        for qid in qids:
            hits = hits_dict[qid]
            ctxs = []
            scores = []

            for h in hits:
                doc = hit2docdict(h)
                ctxs.append(doc["contents"])     
                scores.append(h.score)

            passages_all.append(ctxs)
            scores_all.append(scores)

        return passages_all, scores_all

    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # 검색
        passages_all, scores_all = self.search(queries, qids, k=k)

        list_input_text_without_answer = []

        for question, ctxs in zip(queries, passages_all):
            context_blocks = []
            for c in ctxs:
                lines = c.split('\n')
                title = lines[0] if lines else ""
                passage = "\n".join(lines[1:]) if len(lines) > 1 else ""
                block = f"Title: {title}\nPassage: {passage}"
                context_blocks.append(block)

            input_text_ctx = "\n\n".join(context_blocks)
            input_text_without_answer = f"Question: {question}\n\n{input_text_ctx}"

            # 최종 프롬프트
            prompt = f"{input_text_without_answer}\n\nAnswer: "
            list_input_text_without_answer.append(prompt)
        return list_input_text_without_answer
    
    # 입력을 model에 넣어 생성 수행
    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # 1) qid 기본값 처리
        if qids is None:
            qids = [str(i) for i in range(len(queries))]

        # 2) 검색 + 프롬프트 생성
        prompts = self.make_augmented_inputs_for_generate(
            queries=queries,
            qids=qids,
            k=k,
        )
        
        # 3) 토큰화
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        # outputs.shape = (batch_size, total_seq_len)
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs
