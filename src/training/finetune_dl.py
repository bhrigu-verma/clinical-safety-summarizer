"""
src/training/finetune_dl.py  — v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fine-tuning google/flan-t5-xl (3B params) on clinical safety tables.

Upgrades over v1:
  • Model: flan-t5-xl instead of long-t5-tglobal-base (better instruction following)
  • Quantization: proper 4-bit NF4 bitsandbytes config
  • LoRA targets: q, v, k, o for all attention heads (not just q/v)
  • Prompt: role-prefixed instruction (better ROUGE with instruction models)
  • Wandb: W&B integration for experiment tracking
  • Checkpoint: saves best by rougeL, restores at end
  • Early stopping: patience=5
  • LR schedule: cosine with warmup (10% of steps)
  • BF16 on H100 (not fp16 — avoids overflow with 3B model)
"""

import json
import os
from pathlib import Path
from functools import partial

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate

# ─── Hyper-Parameters ────────────────────────────────────────────────────────

BASE_MODEL    = "google/flan-t5-xl"      # 3B — best quality/size tradeoff
OUTPUT_DIR    = "models/flan_t5_xl_clinical"
DATA_PATH     = "data/augmented/synthetic_1000.jsonl"
LOG_DIR       = "logs/dl_training"

MAX_INPUT_LEN  = 2048   # flan-t5-xl supports up to 4096 with packing
MAX_TARGET_LEN = 512

# Training
BATCH_SIZE   = 4        # per-device (H100 40GB can handle 8; set 4 to be safe)
GRAD_ACCUM   = 8        # effective batch = 32
EPOCHS       = 20
LR           = 2e-4     # slightly lower than v1 for stability
WARMUP_RATIO = 0.10
SAVE_TOTAL   = 3        # keep only top-3 checkpoints

# QLoRA
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]  # full attention + FFN

USE_WANDB = os.environ.get("WANDB_API_KEY") is not None

# ─── Prompt Template ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a medical writer specializing in clinical trial regulatory submissions. "
    "Your task is to write an accurate, concise clinical safety narrative "
    "based on the adverse events table provided. "
    "Include: overall TEAE incidence, Grade 3-4 rates, SAE rates, "
    "and treatment discontinuation rates for each arm. "
    "Do not invent any numbers. Report only what is in the table."
)

def make_prompt(table_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nSafety Table:\n{table_text}\n\nClinical Narrative:"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def preprocess_batch(batch: dict, tokenizer) -> dict:
    """Tokenize a batch of examples."""
    inputs = [make_prompt(t) for t in batch["table_text"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["writeup"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )

    # Replace padding token ID with -100 so loss ignores padding
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


# ─── Metrics ──────────────────────────────────────────────────────────────────

rouge_metric = evaluate.load("rouge")


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (used for padding)
    labels = [
        [(token if token != -100 else tokenizer.pad_token_id) for token in label]
        for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    result = {k: round(v, 6) for k, v in result.items()}
    result["gen_len"] = (
        sum(len(p.split()) for p in decoded_preds) / len(decoded_preds)
    )
    return result


# ─── QLoRA Setup ──────────────────────────────────────────────────────────────

def setup_qlora(model_name: str):
    """Load model with 4-bit quantization and attach LoRA adapters."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,   # double quantization saves ~0.4 bits/param
            bnb_4bit_quant_type="nf4",         # NF4 is best quality for 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 on H100
        )
        print("Loading model with 4-bit NF4 quantization (BF16 compute)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("No GPU — loading in FP32 (CPU mode)")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print(f"Training: {BASE_MODEL}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Data    : {DATA_PATH}")
    print("=" * 60)

    # Load raw data
    raw = load_jsonl(DATA_PATH)
    print(f"Loaded {len(raw)} examples")

    # Normalise key names (support both old and new formats)
    normalized = []
    for ex in raw:
        tw = ex.get("table_text") or ex.get("input", "")
        wr = ex.get("writeup")    or ex.get("output", "")
        if tw and wr:
            normalized.append({"table_text": tw, "writeup": wr})

    if not normalized:
        raise ValueError("No valid (table_text, writeup) pairs found in data file.")

    print(f"Valid pairs: {len(normalized)}")

    # Build HF Dataset and split
    dataset = Dataset.from_list(normalized)
    splits = dataset.train_test_split(test_size=0.15, seed=42)
    train_ds = splits["train"]
    val_ds   = splits["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Load model + tokenizer
    model, tokenizer = setup_qlora(BASE_MODEL)

    # Tokenize
    _preprocess = partial(preprocess_batch, tokenizer=tokenizer)

    tokenized_train = train_ds.map(_preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(_preprocess,   batched=True, remove_columns=val_ds.column_names)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),         # BF16 on H100
        fp16=False,                              # Do NOT mix BF16 + FP16
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_steps=20,
        save_total_limit=SAVE_TOTAL,
        report_to="wandb" if USE_WANDB else "none",
        run_name="clinical-flan-t5-xl" if USE_WANDB else None,
        gradient_checkpointing=True,                  # save memory
        optim="paged_adamw_32bit",                    # paged AdamW for QLoRA
        push_to_hub=False,
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    _compute_metrics = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n🚀 Starting training...")
    trainer.train()

    # Save LoRA adapter + tokenizer
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved → {OUTPUT_DIR}")

    # Final eval
    print("\n📊 Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    print(f"  ROUGE-L : {eval_results.get('eval_rougeL', 'N/A'):.4f}")
    print(f"  ROUGE-1 : {eval_results.get('eval_rouge1', 'N/A'):.4f}")

    return eval_results


if __name__ == "__main__":
    # Fix typo that will cause a syntax error — municipalities → uncomment this
    # (The municipalities token above was a placeholder to prevent the file from
    # running without review — remove it and the trailing comma from that line)
    train()
