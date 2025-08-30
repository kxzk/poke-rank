import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# Constants
LABELS = ["S", "A+", "A", "B", "C", "D"]
LABEL_REGEX = re.compile(r"^(S|A\+|A|B|C|D)\b")
SYSTEM_PROMPT = """You are a Pokemon Pocket (mobile game) TCG card ranker.
Reply with ONE label from this set exactly: S, A+, A, B, C, or D"""

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",
}


def prepare_data(filepath="data.json", val_split=0.2, seed=23):
    """Load and split dataset."""
    raw = load_dataset("json", data_files=filepath, split="train")
    raw = raw.filter(lambda x: x["card_rank"] in LABELS).shuffle(seed=seed)

    n = len(raw)
    val_size = max(1, int(val_split * n))
    return raw.select(range(n - val_size)), raw.select(range(n - val_size, n))


def format_for_training(ds, tokenizer):
    """Convert dataset to chat format."""

    def to_messages(desc, rank):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc},
            {"role": "assistant", "content": rank},
        ]

    def format_batch(batch):
        chats = [
            to_messages(d, r) for d, r in zip(batch["card_desc"], batch["card_rank"])
        ]
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in chats
        ]
        return {"text": texts}

    return ds.map(format_batch, batched=True, remove_columns=ds.column_names)


def setup_model():
    """Initialize model with LoRA adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    return model, tokenizer


def train(model, tokenizer, train_ds, val_ds):
    """Fine-tune the model."""
    train_formatted = format_for_training(train_ds, tokenizer)
    val_formatted = format_for_training(val_ds, tokenizer)

    use_bf16 = torch.cuda.is_bf16_supported()

    config = SFTConfig(
        output_dir="./outputs",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        seed=23,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    trainer.save_model("./card_ranker_lora")
    return model


def evaluate(model, tokenizer, val_ds):
    """Evaluate model accuracy."""
    # Prep for inference
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    def predict(desc):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = {"input_ids": inputs.to(model.device)}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,  # Short output; len("A+")=2 + buffer
                do_sample=False,
                use_cache=False,  # Avoid Unsloth KV cache issues
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        gen = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract label with fallback
        match = LABEL_REGEX.search(gen)
        if match:
            return match.group(1)
        return gen.split()[0].strip() if gen else "UNK"

    # Run evaluation
    correct = 0
    predictions, actuals = [], []

    for row in tqdm(val_ds, desc="Evaluating"):
        pred = predict(row["card_desc"])
        actual = row["card_rank"]

        predictions.append(pred)
        actuals.append(actual)
        correct += pred == actual

    accuracy = correct / len(val_ds)
    print(f"\nValidation Accuracy: {accuracy:.3f} ({correct}/{len(val_ds)})")
    return predictions, actuals, accuracy


def main():
    # Setup
    train_ds, val_ds = prepare_data()
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model, tokenizer = setup_model()

    # Train
    print("\nTraining...")
    model = train(model, tokenizer, train_ds, val_ds)

    # Evaluate
    print("\nEvaluating...")
    preds, golds, acc = evaluate(model, tokenizer, val_ds)

    # Optional: confusion matrix
    from collections import Counter

    errors = [(g, p) for g, p in zip(golds, preds) if g != p]
    if errors:
        print(f"\nTop errors: {Counter(errors).most_common(5)}")


if __name__ == "__main__":
    main()
