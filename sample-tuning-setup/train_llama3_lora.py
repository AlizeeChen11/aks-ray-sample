import argparse
import os
from pathlib import Path

import pandas as pd
import ray
import ray.data
import ray.train as train
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ray.air.config import RunConfig, ScalingConfig
from ray.train.huggingface.transformers._transformers_utils import (
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.utils.logging import disable_progress_bar, enable_progress_bar


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET_ID = "tatsu-lab/alpaca"
DEFAULT_OUTPUT_DIR = "/results/llama3-lora"


def main(args):
    print("Initializing Ray...")
    ray.init()

    if args.download_model_on_each_node:
        print("Checking gated model access and warming model metadata on each Ray node...")
        run_on_every_node(download_model_metadata, model_id=args.model_id)

    print("Loading dataset...")
    current_dataset = load_training_dataset(args)
    split_name = args.dataset_split
    validation_split_name = args.validation_split

    if validation_split_name and validation_split_name in current_dataset:
        validation_dataset = current_dataset[validation_split_name]
    elif "test" in current_dataset:
        validation_dataset = current_dataset["test"]
    else:
        split_dataset = current_dataset[split_name].train_test_split(
            test_size=args.validation_size,
            seed=42,
        )
        current_dataset = split_dataset
        split_name = "train"
        validation_dataset = current_dataset["test"]

    ray_datasets = {
        "train": ray.data.from_huggingface(current_dataset[split_name]),
        "validation": ray.data.from_huggingface(validation_dataset),
    }

    tokenize_batch = make_tokenize_batch(args.model_id, args.max_length)
    processed_datasets = {
        dataset_name: dataset.map_batches(
            format_instruction_batch,
            batch_format="pandas",
        ).map_batches(
            tokenize_batch,
            batch_format="pandas",
        )
        for dataset_name, dataset in ray_datasets.items()
    }

    if args.max_train_samples > 0:
        processed_datasets["train"] = processed_datasets["train"].limit(args.max_train_samples)
    if args.max_eval_samples > 0:
        processed_datasets["validation"] = processed_datasets["validation"].limit(args.max_eval_samples)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "bf16": args.bf16,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_rank": args.lora_rank,
            "max_steps": args.max_steps,
            "model_id": args.model_id,
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "save_steps": args.save_steps,
            "use_qlora": args.use_qlora,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": args.cpus_per_worker, "GPU": 1},
        ),
        datasets=processed_datasets,
        run_config=RunConfig(storage_path=args.output_dir),
    )

    print("Running trainer.fit()...")
    trainer.fit()


def load_training_dataset(args):
    if args.dataset_path:
        data_files = {args.dataset_split: args.dataset_path}
        extension = Path(args.dataset_path).suffix.lower()
        if extension in [".json", ".jsonl"]:
            return load_dataset("json", data_files=data_files)
        if extension == ".csv":
            return load_dataset("csv", data_files=data_files)
        raise ValueError("dataset_path must point to a .json, .jsonl, or .csv file")

    return load_dataset(args.dataset_id)


def format_instruction_batch(batch: pd.DataFrame) -> pd.DataFrame:
    texts = []
    for row in batch.to_dict(orient="records"):
        if row.get("text"):
            texts.append(str(row["text"]))
            continue

        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        response = str(row.get("output", row.get("response", ""))).strip()

        user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
        texts.append(
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )

    return pd.DataFrame({"text": texts})


def make_tokenize_batch(model_id, max_length):
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch: pd.DataFrame) -> dict:
        tokenized = tokenizer(
            list(batch["text"]),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return dict(tokenized)

    return tokenize_batch


def force_on_node(node_id, remote_func):
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    return ray.remote(remote_func).options(scheduling_strategy=scheduling_strategy)


def run_on_every_node(remote_func, **remote_kwargs):
    refs = []
    for node in ray.nodes():
        if node["Alive"]:
            remote_on_node = force_on_node(node["NodeID"], remote_func)
            refs.append(remote_on_node.remote(**remote_kwargs))
    return ray.get(refs)


def download_model_metadata(model_id):
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Checking access to {model_id}...")
    AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    AutoConfig.from_pretrained(model_id, token=hf_token)
    print(f"Access check for {model_id} succeeded")
    return True


def train_func(config):
    ctx = train.get_context()
    rank = ctx.get_world_rank()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is required for gated Llama 3 models")

    torch.backends.cuda.matmul.allow_tf32 = True
    disable_progress_bar()

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], token=hf_token, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[Worker {rank}] Loading {config['model_id']}...")
    model_kwargs = {
        "token": hf_token,
        "torch_dtype": torch.bfloat16 if config["bf16"] else torch.float16,
        "use_cache": False,
    }

    if config["use_qlora"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config["model_id"], **model_kwargs)
    model.config.use_cache = False

    if config["use_qlora"]:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_dataset = train.get_dataset_shard("train")
    eval_dataset = train.get_dataset_shard("validation")
    train_iterable = train_dataset.iter_torch_batches(
        batch_size=config["per_device_train_batch_size"],
        local_shuffle_buffer_size=config["per_device_train_batch_size"] * 8,
    )
    eval_iterable = eval_dataset.iter_torch_batches(batch_size=config["per_device_train_batch_size"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        save_strategy="no",
        logging_steps=10,
        bf16=config["bf16"],
        fp16=not config["bf16"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_on_each_node=False,
        report_to="none",
        disable_tqdm=True,
        push_to_hub=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    hf_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_iterable,
        eval_dataset=eval_iterable,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    hf_trainer = prepare_trainer(hf_trainer)
    hf_trainer.train()

    if hf_trainer.is_world_process_zero():
        hf_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved LoRA adapter and tokenizer to {output_dir}")

    enable_progress_bar()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 with Ray Train and LoRA")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument("--validation_size", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--cpus_per_worker", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--download_model_on_each_node", action="store_true")
    parser.set_defaults(bf16=True)
    main(parser.parse_args())
