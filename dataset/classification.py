import os
import datasets
import itertools
import functools
import torch
from torch.nn.utils.rnn import pad_sequence

# 텍스트 분류 데이터셋을 불러오고 전처리하여 train/val으로 준비
def prepare_classification_dataset(tokenizer,
                             dataset_name_or_path: str = "SetFit/20_newsgroups",
                             dataset_subset: str = None,
                             text_column_name: str = "text",
                             label_column_name: str = "label",
                             train_sample_size: int =-1,
                             cache_path:str="cache"):
    # 캐시된 전처리 데이터셋이 있다면 불러와서 사용
    if cache_path is not None and os.path.exists(os.path.join(cache_path,"classification")):
        print(f"Using pre-downloaded dataset from {cache_path}.")
        train = datasets.load_from_disk(os.path.join(cache_path,"classification","train"))
        validation = datasets.load_from_disk(os.path.join(cache_path,"classification","eval"))
        return train, validation
    # 없다면 다운로드 후 전처리 (기본값: "SetFit/20_newsgroups")
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
        
        train = dataset["train"] if train_sample_size == -1 else dataset["train"].select(range(train_sample_size))
        validation = dataset[validation_split]
        # 전처리 함수 적용 (partial 함수)
        processing_lambda = functools.partial(
            _preprocessing,
            tokenizer=tokenizer,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )
        # train에 _preprocessing 적용
        train = train.map(
            processing_lambda,
            batched=True,
            remove_columns=train.column_names,
            num_proc=4,
            desc="Tokenizing",
        )
        # val에 _preprocessing 적용
        validation = validation.map(
            processing_lambda,
            batched=True,
            remove_columns=validation.column_names,
            num_proc=4,
            desc="Tokenizing",
        )

        if cache_path is not None:
            os.makedirs(os.path.join(cache_path,"classification"), exist_ok=True)
            print(f"Saving dataset to {cache_path}...")
            train.save_to_disk(os.path.join(cache_path,"classification","train"), max_shard_size="500MB")
            validation.save_to_disk(os.path.join(cache_path,"classification","eval"), max_shard_size="500MB")
        return train, validation

# 한 batch의 문장을 tokenizer로 변환
def _preprocessing(examples, tokenizer, text_column_name, label_column_name):
    text = examples[text_column_name]
    labels = examples[label_column_name]

    inputs = tokenizer(text, add_special_tokens=True, padding="longest", return_tensors="pt",max_length=512, truncation=True)
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return inputs

# 서로 길이가 다른 input들을 pad 처리해서 하나의 batch로 묶음.
# input_ids, attention_mask, labels를 딕셔너리 형태로 반환
def collate_fn_for_classification(batch, tokenizer,):

    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    return inputs