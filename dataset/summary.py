import os
import datasets
import itertools
import functools
import torch

# Summarization에 사용할 데이터셋을 불러오고 전처리해서 변환
def prepare_summary_dataset(tokenizer,
                             dataset_name_or_path: str = "abisee/cnn_dailymail",
                             dataset_subset: str = "3.0.0",
                             context_max_length: int = 896,
                             target_max_length: int = 128,
                             context_column_name: str = "article",
                             target_column_name: str = "highlights",
                             prompt: str = "Context: {context}\n Summary:\n",
                             train_sample_size: int =-1,
                             cache_path:str="cache"):
    # 캐시 경로에 데이터가 존재하는지 확인
    if cache_path is not None and os.path.exists(os.path.join(cache_path,"summary")):
        # 캐시 데이터셋 로드
        print(f"Using pre-downloaded dataset from {cache_path}.")
        train = datasets.load_from_disk(os.path.join(cache_path,"summary","train"))
        validation = datasets.load_from_disk(os.path.join(cache_path,"summary","eval"))
        return train, validation
    # 캐시가 없을 경우 -> 다운로드 시작 / 검증 데이터 split 결정
    else:
        print(f"Download and prerpocessing dataset from {dataset_name_or_path} on subset {dataset_subset}...")
        dataset = datasets.load_dataset(dataset_name_or_path, dataset_subset)
        if "eval" in dataset:
            validation_split = "eval"
        elif "test" in dataset:
            validation_split = "test"
        else:
            print("Using train split for validation.")
            dataset = datasets.train_test_split(test_size=0.1)
            validation_split = "test"
        # 데이터셋 불러오기
        train = dataset["train"] if train_sample_size == -1 else dataset["train"].select(range(train_sample_size))
        validation = dataset[validation_split]
        
        # 전처리 함수에 인자 고정 (partial 함수 생성) / _preprocess 함수에 tokenizer등 인자들을 미리 고정
        processing_lambda = functools.partial(
            _preprocess,
            tokenizer=tokenizer,
            context_column_name=context_column_name,
            target_column_name=target_column_name,
            context_max_length=context_max_length,
            target_max_length=target_max_length,
            prompt=prompt
        )
        # train에 _preprocess 함수 적용
        train = train.map(
            processing_lambda,
            batched=True,
            remove_columns=train.column_names,
            num_proc=1,
            batch_size=64,
            desc="Preprocessing",
            load_from_cache_file=False
        )
        # val에 _preprocess 함수 적용
        validation = validation.map(
            processing_lambda,
            batched=True,
            remove_columns=validation.column_names,
            num_proc=1,
            batch_size=64,
            desc="Preprocessing",
            load_from_cache_file=False
        )

        # 캐시로 저장
        if cache_path is not None:
            os.makedirs(os.path.join(cache_path,"summary"), exist_ok=True)
            print(f"Saving dataset to {cache_path}...")
            train.save_to_disk(os.path.join(cache_path,"summary","train"), max_shard_size="500MB")
            validation.save_to_disk(os.path.join(cache_path,"summary","eval"), max_shard_size="500MB")
        return train, validation

# context: 기사 원문, target: 요약 정답
def _preprocess(examples, tokenizer, context_column_name, target_column_name, context_max_length, target_max_length, prompt):
    # 문장을 prompt 형태로 포맷 / truncation을 위해서 토큰화
    contexts = tokenizer(examples[context_column_name], add_special_tokens=False, truncation=True, max_length=context_max_length-10)
    targets = tokenizer(examples[target_column_name], add_special_tokens=False, truncation=True, max_length=target_max_length)
    # prompt 형태로 포맷한 문장을 tokenizer에 넣음 / 토큰화
    contexts = tokenizer.batch_decode(contexts["input_ids"], skip_special_tokens=True)
    targets = tokenizer.batch_decode(targets["input_ids"], skip_special_tokens=True)

    # Prompt 형태로 input 만들기 / 고정된 길이로 만듦
    # prompt = "Context: {context}\n Summary:" -> 이 input을 이용해서 model이 예측값을 생성하고 그것과 label을 비교하여 loss 계산
    inputs = tokenizer([prompt.format(context=c) for c in contexts],add_special_tokens = False,padding=False,truncation=True)
    labels = tokenizer(targets,add_special_tokens = False,padding = False,truncation=True)

    pad_id = tokenizer.pad_token_id

    def _pad(seq, max_len, pad):
        assert len(seq) <= max_len, "시퀀스 길이가 max_len 초과"
        return seq + [pad] * (max_len - len(seq))
 
    for i in range(len(inputs['input_ids'])):
        total_len = 1024
        target_max_length = 192

        target_ids = labels['input_ids'][i][:target_max_length] # 토큰화 된 정답 생성
        prompt_ids = tokenizer(prompt.format(context=""), add_special_tokens=False)["input_ids"] # 토큰화 된 프롬프트 생성

        max_context_len = total_len - len(target_ids) - len(prompt_ids) # 최대 길이 계산
        context_ids = inputs['input_ids'][i][:max_context_len] # 토큰화 된 질문 생성

        # prompt + context 다시 구성
        prompt_text = prompt.format(context=tokenizer.decode(context_ids, skip_special_tokens=True)) # 프롬프트 형식으로 변경
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, padding=False, truncation=True)["input_ids"] # 다시 토큰화
        context_ids = prompt_ids  # prompt 포함된 context

        # input_ids = context + target
        input_ids = context_ids + target_ids
        inputs['input_ids'][i] = _pad(input_ids, total_len, pad_id)

        # 본문 1, 패딩 0
        inputs['attention_mask'][i] = _pad([1] * len(input_ids), total_len, 0)

        # labels: 본문 -100, 패딩 -100, 요약토큰 (=정답토큰)
        labels['input_ids'][i] = _pad(([-100] * len(context_ids)) + target_ids, total_len, -100)

    inputs["labels"] = labels["input_ids"]
    return inputs

def collate_fn_for_summary(batch, tokenizer, pad_to_multiple_of=1024):
    input_ids = [example["input_ids"] for example in batch] # 모델이 입력으로 받을 문장의 토큰 ID
    attention_mask = [example["attention_mask"] for example in batch] # 실제 단어 vs pad 여부 (1 vs 0)
    labels = [example["labels"] for example in batch] # 정답 텍스트 토큰 ID (pad는 -100으로 마스킹)

    # 형태 변환
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    max_seq_length = attention_mask.eq(1).sum(-1).max().item()
    if max_seq_length % pad_to_multiple_of != 0:
        max_seq_length = (max_seq_length // pad_to_multiple_of + 1) * pad_to_multiple_of

    if tokenizer.padding_side == "left":
        input_ids = input_ids[:, -max_seq_length:]
        attention_mask = attention_mask[:, -max_seq_length:]
        labels = labels[:, -max_seq_length:]
    else:
        input_ids = input_ids[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]
        labels = labels[:, :max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
