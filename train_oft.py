#!/usr/bin/env python3
"""
OFT (Orthogonal Fine-Tuning) for Qwen2.5-1.5B on SST-2 Sentiment Analysis
============================================================================
This script fine-tunes a pretrained Qwen2.5-1.5B model using OFT (Orthogonal
Fine-Tuning) from HuggingFace PEFT for binary sentiment classification on the
SST-2 dataset.

Usage:
    python train_oft.py [--num_epochs 3] [--batch_size 8] [--lr 1e-4]
"""

import os
import json
import argparse
import time
from datetime import datetime

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import OFTConfig, get_peft_model
from tqdm import tqdm


# ===========================================================================
# Configuration
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="OFT Fine-tuning for Sentiment Analysis"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B",
        help="Pretrained model name",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--oft_r", type=int, default=8, help="OFT rank (number of OFT blocks)")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ===========================================================================
# Prompt template
# ===========================================================================
PROMPT_TEMPLATE = (
    "Classify the sentiment of the following movie review as "
    "'positive' or 'negative'.\n\n"
    "Review: {text}\n\n"
    "Sentiment: "
)


# ===========================================================================
# Custom Dataset
# ===========================================================================
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example["sentence"]
        label = "positive" if example["label"] == 1 else "negative"
        prompt = PROMPT_TEMPLATE.format(text=text)

        # Tokenize prompt and answer SEPARATELY, then concatenate
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(label, add_special_tokens=False)
        eos_ids = [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids + eos_ids
        # Labels: -100 for prompt tokens, actual ids for answer + eos
        labels = [-100] * len(prompt_ids) + answer_ids + eos_ids

        # Truncate if needed
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "answer": label,
            "prompt": prompt,
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    answers = []
    prompts = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        answers.append(item["answer"])
        prompts.append(item["prompt"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "answers": answers,
        "prompts": prompts,
    }


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_model(model, tokenizer, eval_data, device, max_new_tokens=5):
    """Evaluate model by generating predictions and computing accuracy."""
    model.eval()
    predictions = []
    references = []

    for example in tqdm(eval_data, desc="Evaluating"):
        text = example["sentence"]
        label = "positive" if example["label"] == 1 else "negative"
        prompt = PROMPT_TEMPLATE.format(text=text)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.strip().lower()

        if "positive" in generated_text:
            pred = "positive"
        elif "negative" in generated_text:
            pred = "negative"
        else:
            pred = generated_text

        predictions.append(pred)
        references.append(label)

    correct = sum(p == r for p, r in zip(predictions, references))
    accuracy = correct / len(references) if references else 0

    return {
        "accuracy": accuracy,
        "total": len(references),
        "correct": correct,
        "predictions": predictions,
        "references": references,
    }


# ===========================================================================
# Plot
# ===========================================================================
def plot_training_loss(train_losses, eval_losses, output_dir):
    """Plot training and evaluation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        range(1, len(train_losses) + 1), train_losses,
        "b-", linewidth=1, alpha=0.7, label="Training Loss (per step)"
    )

    if eval_losses:
        # Place eval loss markers at end of each epoch
        steps_per_epoch = len(train_losses) // len(eval_losses)
        eval_x = [(i + 1) * steps_per_epoch for i in range(len(eval_losses))]
        ax.plot(eval_x, eval_losses, "r-s", linewidth=2, markersize=8, label="Eval Loss (per epoch)")

    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("OFT Fine-tuning: Training Loss Curve", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150)
    plt.close()
    print(f"Training loss plot saved to {output_dir}/training_loss.png")


# ===========================================================================
# Training loop
# ===========================================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    step_losses = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        total_loss += loss_val
        step_losses.append(loss_val)

        pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, step_losses


@torch.no_grad()
def compute_eval_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print(f"Model loaded. Total parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Loading SST-2 dataset...")
    print(f"{'='*60}")

    dataset = load_dataset("glue", "sst2")
    train_raw = dataset["train"].shuffle(seed=args.seed)
    eval_raw = dataset["validation"]

    if args.train_samples > 0 and args.train_samples < len(train_raw):
        train_raw = train_raw.select(range(args.train_samples))
    if args.eval_samples > 0 and args.eval_samples < len(eval_raw):
        eval_raw = eval_raw.select(range(args.eval_samples))

    print(f"Train samples: {len(train_raw)}")
    print(f"Eval samples:  {len(eval_raw)}")

    # ------------------------------------------------------------------
    # 3. Evaluate BASE model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating BASE model (before OFT fine-tuning)...")
    print(f"{'='*60}")

    base_results = evaluate_model(model, tokenizer, eval_raw, device)
    print(f"\n>>> Base model accuracy: {base_results['accuracy']:.4f} "
          f"({base_results['correct']}/{base_results['total']})")

    with open(os.path.join(args.output_dir, "base_model_results.json"), "w") as f:
        json.dump({
            "accuracy": base_results["accuracy"],
            "correct": base_results["correct"],
            "total": base_results["total"],
        }, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Apply OFT via PEFT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Applying OFT (r={args.oft_r})...")
    print(f"{'='*60}")

    oft_config = OFTConfig(
        r=args.oft_r,
        target_modules=["q_proj", "v_proj"],
        module_dropout=0.0,
        init_weights=True,
        oft_block_size=0,
    )

    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    # Verify tokenization works correctly
    sample_ds = SentimentDataset([train_raw[0]], tokenizer, args.max_len)
    sample = sample_ds[0]
    valid_labels = (sample["labels"] != -100).sum().item()
    print(f"\n[Sanity Check] Sample input length: {len(sample['input_ids'])}")
    print(f"[Sanity Check] Valid label tokens: {valid_labels}")
    print(f"[Sanity Check] Answer: '{sample['answer']}'")
    assert valid_labels > 0, "ERROR: All labels are masked! Tokenization bug."

    # ------------------------------------------------------------------
    # 5. Create DataLoaders
    # ------------------------------------------------------------------
    train_dataset = SentimentDataset(train_raw, tokenizer, args.max_len)
    eval_dataset = SentimentDataset(eval_raw, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------
    # 6. Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ------------------------------------------------------------------
    # 7. Training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting OFT fine-tuning...")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup:      {warmup_steps}")
    print(f"  LR:          {args.lr}")
    print(f"{'='*60}")

    all_step_losses = []
    epoch_train_losses = []
    epoch_eval_losses = []
    start_time = time.time()

    for epoch in range(args.num_epochs):
        avg_train_loss, step_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        all_step_losses.extend(step_losses)
        epoch_train_losses.append(avg_train_loss)

        e_loss = compute_eval_loss(model, eval_loader, device)
        epoch_eval_losses.append(e_loss)

        print(f"\nEpoch {epoch+1}/{args.num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Eval Loss={e_loss:.4f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")

    # Save loss curve
    plot_training_loss(all_step_losses, epoch_eval_losses, args.output_dir)

    # Save adapter
    adapter_path = os.path.join(args.output_dir, "oft_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"OFT adapter saved to {adapter_path}")

    # ------------------------------------------------------------------
    # 8. Evaluate fine-tuned model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating FINE-TUNED model (after OFT)...")
    print(f"{'='*60}")

    ft_results = evaluate_model(model, tokenizer, eval_raw, device)
    print(f"\n>>> Fine-tuned accuracy: {ft_results['accuracy']:.4f} "
          f"({ft_results['correct']}/{ft_results['total']})")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    improvement = ft_results["accuracy"] - base_results["accuracy"]

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Base model accuracy:       {base_results['accuracy']:.4f}")
    print(f"  Fine-tuned model accuracy: {ft_results['accuracy']:.4f}")
    print(f"  Improvement:               {improvement:+.4f}")
    print(f"  Training time:             {training_time:.1f}s")
    print(f"  Trainable params:          {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Qualitative examples
    print(f"\n{'='*60}")
    print("QUALITATIVE EXAMPLES (first 10)")
    print(f"{'='*60}")

    qualitative = []
    for i in range(min(20, len(eval_raw))):
        text = eval_raw[i]["sentence"]
        ref = "positive" if eval_raw[i]["label"] == 1 else "negative"
        base_pred = base_results["predictions"][i]
        ft_pred = ft_results["predictions"][i]
        qualitative.append({
            "sentence": text,
            "ground_truth": ref,
            "base_prediction": base_pred,
            "finetuned_prediction": ft_pred,
        })
        if i < 10:
            status = "✓" if ft_pred == ref else "✗"
            print(f"\n  [{i+1}] {text[:80]}...")
            print(f"      GT: {ref} | Base: {base_pred} | OFT: {ft_pred} {status}")

    # Save all results
    results = {
        "model": args.model_name,
        "method": "OFT",
        "oft_rank": args.oft_r,
        "target_modules": ["q_proj", "v_proj"],
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_raw),
        "eval_samples": len(eval_raw),
        "training_time_seconds": round(training_time, 1),
        "base_accuracy": base_results["accuracy"],
        "finetuned_accuracy": ft_results["accuracy"],
        "improvement": round(improvement, 4),
        "final_train_loss": epoch_train_losses[-1],
        "final_eval_loss": epoch_eval_losses[-1],
        "all_step_losses": all_step_losses,
        "epoch_train_losses": epoch_train_losses,
        "epoch_eval_losses": epoch_eval_losses,
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.output_dir, "qualitative_examples.json"), "w") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE! All outputs saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
