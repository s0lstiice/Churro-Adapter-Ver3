#!/usr/bin/env python3
"""Memory-conscious full-page QLoRA fine-tuning for CHURRO/Qwen2.5-VL."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from progress_client import ProgressTask


MODEL_ID = "stanford-oval/churro-3B"
SYSTEM_PROMPT = (
    "Transcribe every visible word in this entire historical document to XML. "
    "Preserve reading order, spelling, capitalization, and punctuation. Never summarize, "
    "shorten, skip, or replace visible text with comments or placeholders such as "
    "'omitted for brevity'. Continue through the bottom of the page and close the XML only "
    "after all visible text has been transcribed."
)
LINE_SYSTEM_PROMPT = (
    "Transcribe this single handwritten line exactly. Preserve the visible spelling, "
    "capitalization, and punctuation. Return only the transcription."
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def target_xml(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = "\n".join(f"        <Line>{html.escape(line)}</Line>" for line in lines)
    return (
        '<HistoricalDocument xmlns="http://example.com/historicaldocument">\n'
        "  <Page>\n"
        "    <Body>\n"
        "      <Paragraph>\n"
        f"{body}\n"
        "      </Paragraph>\n"
        "    </Body>\n"
        "  </Page>\n"
        "</HistoricalDocument>"
    )


def image_content(path: str, max_pixels: int) -> dict:
    return {
        "type": "image",
        "image": f"file://{Path(path).resolve()}",
        "min_pixels": 256 * 28 * 28,
        "max_pixels": max_pixels,
    }


def task_type(row: dict) -> str:
    value = str(row.get("task_type") or row.get("granularity") or "page").strip().lower()
    if value in {"full_page", "full-page", "document"}:
        value = "page"
    if value not in {"page", "line"}:
        raise ValueError(f"{row.get('id', '<unknown>')} has unsupported task_type={value!r}")
    return value


def target_for(row: dict) -> str:
    if task_type(row) == "line":
        # Do not normalize case or punctuation: official transcript surfaces are
        # the supervision signal for mixed-scale recognition.
        return str(row["text"]).strip()
    return target_xml(row["text"])


def messages_for(row: dict, max_pixels: int, include_answer: bool) -> list[dict]:
    line_task = task_type(row) == "line"
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": LINE_SYSTEM_PROMPT if line_task else SYSTEM_PROMPT}],
        },
        {"role": "user", "content": [image_content(row["image"], max_pixels)]},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_for(row)}]}
        )
    return messages


def make_batch(processor, row: dict, max_pixels: int, max_sequence_length: int, device):
    prompt_messages = messages_for(row, max_pixels, include_answer=False)
    full_messages = messages_for(row, max_pixels, include_answer=True)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(full_messages)
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    length = int(full.input_ids.shape[1])
    prompt_length = int(prompt.input_ids.shape[1])
    if length > max_sequence_length:
        raise ValueError(
            f"{row['id']} needs {length} tokens, above --max-sequence-length {max_sequence_length}"
        )
    if not torch.equal(full.input_ids[:, :prompt_length], prompt.input_ids):
        raise RuntimeError(f"chat-template prefix mismatch for {row['id']}")
    labels = full.input_ids.clone()
    labels[:, :prompt_length] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[full.input_ids == pad_id] = -100
    full["labels"] = labels
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in full.items()}


def load_model(model_id: str, lora_rank: int, lora_alpha: int, adapter: Path | None = None):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(str(adapter) if adapter else model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={"": 0},
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    return processor, model


def pixels_for(row: dict, args) -> int:
    return args.line_max_pixels if task_type(row) == "line" else args.page_max_pixels


@torch.no_grad()
def validation_loss(model, processor, rows, args, task, completed: int) -> tuple[float, dict, int]:
    model.eval()
    losses = []
    losses_by_task = {"line": [], "page": []}
    skipped_overlength = 0
    for row in rows:
        try:
            batch = make_batch(
                processor, row, pixels_for(row, args), args.max_sequence_length, model.device
            )
        except ValueError as error:
            if "above --max-sequence-length" not in str(error):
                raise
            skipped_overlength += 1
            completed += 1
            task.update(
                completed,
                message=f"validation skipped overlength: {row['id']}",
                metrics={
                    "validation_loss_running": sum(losses) / max(1, len(losses)),
                    "validation_skipped_overlength": skipped_overlength,
                },
            )
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
        losses_by_task[task_type(row)].append(losses[-1])
        completed += 1
        task.update(
            completed,
            message=f"validation: {row['id']}",
            metrics={
                "validation_loss_running": sum(losses) / len(losses),
                "validation_line_loss_running": sum(losses_by_task["line"]) / max(1, len(losses_by_task["line"])),
                "validation_page_loss_running": sum(losses_by_task["page"]) / max(1, len(losses_by_task["page"])),
            },
        )
        del batch, loss
    model.train()
    task_means = {
        name: sum(values) / len(values)
        for name, values in losses_by_task.items()
        if values
    }
    # Give line and page behavior equal influence on checkpoint selection even
    # when their validation-set row counts differ.
    balanced = sum(task_means.values()) / max(1, len(task_means))
    return balanced, task_means, completed


def save_checkpoint(model, processor, output: Path, epoch: int, metrics: dict) -> Path:
    checkpoint = output / f"checkpoint-epoch-{epoch:02d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    (checkpoint / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint


def save_page_checkpoint(
    model, processor, output: Path, pages_seen: int, metrics: dict, remaining_rows: list[dict]
) -> Path:
    checkpoint = output / f"checkpoint-page-{pages_seen:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    (checkpoint / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (checkpoint / "remaining_train.jsonl").open("w", encoding="utf-8") as handle:
        for row in remaining_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "latest_page_checkpoint.txt").write_text(
        str(checkpoint.resolve()) + "\n", encoding="utf-8"
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("churro_loc_fullpage_dataset_v1/train.jsonl"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("churro_loc_fullpage_dataset_v1/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("churro_loc_fullpage_qlora_v1"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path, help="Continue training an existing PEFT adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--starting-epoch",
        type=int,
        default=1,
        help="Epoch number assigned to the first epoch in this invocation.",
    )
    parser.add_argument(
        "--initial-best-validation-loss",
        type=float,
        help="Best validation loss from the adapter being continued; enables true resume-aware early stopping.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--line-max-pixels", type=int)
    parser.add_argument("--page-max-pixels", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--max-target-words", type=int)
    parser.add_argument("--max-target-characters", type=int)
    parser.add_argument(
        "--save-every-pages",
        type=int,
        default=250,
        help="Save a resumable adapter and remaining-row manifest every N successful pages; 0 disables.",
    )
    args = parser.parse_args()

    args.line_max_pixels = args.line_max_pixels or args.max_pixels
    args.page_max_pixels = args.page_max_pixels or args.max_pixels

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CHURRO QLoRA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    prefilter = lambda row: (
        (args.max_target_words is None or len(row.get("text", "").split()) <= args.max_target_words)
        and (
            args.max_target_characters is None
            or len(row.get("text", "")) <= args.max_target_characters
        )
    )
    original_train_pages = len(train_rows)
    original_validation_pages = len(validation_rows)
    train_rows = [row for row in train_rows if prefilter(row)]
    validation_rows = [row for row in validation_rows if prefilter(row)]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_validation:
        validation_rows = validation_rows[: args.limit_validation]

    train_task_counts = {
        name: sum(task_type(row) == name for row in train_rows) for name in ("line", "page")
    }
    validation_task_counts = {
        name: sum(task_type(row) == name for row in validation_rows) for name in ("line", "page")
    }
    mixed_scale = all(train_task_counts.values())

    task = ProgressTask(
        "CHURRO mixed-scale QLoRA" if mixed_scale else "CHURRO full-page QLoRA v1",
        total=args.epochs * (len(train_rows) + len(validation_rows)),
        unit="example passes" if mixed_scale else "page passes",
        task_id=("churro-mixed-scale-qlora-" if mixed_scale else "churro-fullpage-qlora-") + args.output.name,
        output_dir=args.output,
        metadata={
            "model": args.model,
            "train_pages": len(train_rows),
            "validation_pages": len(validation_rows),
            "prefiltered_train_pages": original_train_pages - len(train_rows),
            "prefiltered_validation_pages": original_validation_pages - len(validation_rows),
            "max_pixels": args.max_pixels,
            "line_max_pixels": args.line_max_pixels,
            "page_max_pixels": args.page_max_pixels,
            "train_task_counts": train_task_counts,
            "validation_task_counts": validation_task_counts,
        },
    )
    completed = 0
    try:
        task.update(0, message="loading quantized CHURRO")
        processor, model = load_model(args.model, args.lora_rank, args.lora_alpha, args.adapter)
        trainable, total = model.get_nb_trainable_parameters()
        model.print_trainable_parameters()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer_steps_per_epoch = math.ceil(len(train_rows) / args.gradient_accumulation)
        total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, round(total_optimizer_steps * args.warmup_ratio)),
            num_training_steps=total_optimizer_steps,
        )
        run_config = {
            **vars(args),
            "train_manifest": str(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "output": str(args.output),
            "train_pages": len(train_rows),
            "validation_pages": len(validation_rows),
            "original_train_pages": original_train_pages,
            "original_validation_pages": original_validation_pages,
            "prefiltered_train_pages": original_train_pages - len(train_rows),
            "prefiltered_validation_pages": original_validation_pages - len(validation_rows),
            "train_task_counts": train_task_counts,
            "validation_task_counts": validation_task_counts,
            "mixed_scale": mixed_scale,
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
        (args.output / "run_config.json").write_text(
            json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
        )

        history = []
        best_loss = (
            float(args.initial_best_validation_loss)
            if args.initial_best_validation_loss is not None
            else float("inf")
        )
        best_checkpoint = args.adapter if args.initial_best_validation_loss is not None else None
        bad_epochs = 0
        optimizer.zero_grad(set_to_none=True)
        final_epoch = args.starting_epoch + args.epochs - 1
        for epoch in range(args.starting_epoch, final_epoch + 1):
            random.Random(args.seed + epoch).shuffle(train_rows)
            model.train()
            running_loss = 0.0
            successful_pages = 0
            skipped_overlength = 0
            for index, row in enumerate(train_rows, start=1):
                try:
                    batch = make_batch(
                        processor, row, pixels_for(row, args), args.max_sequence_length, model.device
                    )
                except ValueError as error:
                    if "above --max-sequence-length" not in str(error):
                        raise
                    skipped_overlength += 1
                    completed += 1
                    task.update(
                        completed,
                        message=f"epoch {epoch}/{final_epoch}: skipped overlength {row['id']}",
                        metrics={
                            "epoch": epoch,
                            "train_loss_running": running_loss / max(1, successful_pages),
                            "skipped_overlength": skipped_overlength,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                        },
                    )
                    continue
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                raw_loss = float(loss.detach().cpu())
                (loss / args.gradient_accumulation).backward()
                running_loss += raw_loss
                successful_pages += 1
                should_step = successful_pages % args.gradient_accumulation == 0
                if should_step:
                    clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                completed += 1
                task.update(
                    completed,
                    message=f"epoch {epoch}/{final_epoch} [{task_type(row)}]: {row['id']}",
                    metrics={
                        "epoch": epoch,
                        "train_loss_running": running_loss / successful_pages,
                        "skipped_overlength": skipped_overlength,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    },
                )
                del batch, loss

                if args.save_every_pages and successful_pages % args.save_every_pages == 0:
                    save_page_checkpoint(
                        model,
                        processor,
                        args.output,
                        successful_pages,
                        {
                            "epoch": epoch,
                            "manifest_rows_seen": index,
                            "successful_pages": successful_pages,
                            "skipped_overlength": skipped_overlength,
                            "train_loss_running": running_loss / successful_pages,
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        train_rows[index:],
                    )

            if successful_pages % args.gradient_accumulation:
                clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            val_loss, validation_task_losses, completed = validation_loss(
                model, processor, validation_rows, args, task, completed
            )
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, successful_pages),
                "trained_pages": successful_pages,
                "skipped_overlength": skipped_overlength,
                "validation_loss": val_loss,
                "validation_task_losses": validation_task_losses,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(epoch_metrics)
            checkpoint = save_checkpoint(model, processor, args.output, epoch, epoch_metrics)
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_checkpoint = checkpoint
                bad_epochs = 0
                (args.output / "best_checkpoint.txt").write_text(
                    str(checkpoint.resolve()) + "\n", encoding="utf-8"
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.early_stopping_patience:
                    break

        if best_checkpoint is None:
            raise RuntimeError("training produced no checkpoint")
        summary = {
            "best_checkpoint": str(best_checkpoint.resolve()),
            "best_validation_loss": best_loss,
            "history": history,
            "stopped_after_epochs": len(history),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        best_epoch_match = re.search(r"checkpoint-epoch-(\d+)$", best_checkpoint.name)
        task.finish(
            "training complete",
            metrics={
                "best_validation_loss": best_loss,
                "best_epoch": int(best_epoch_match.group(1)) if best_epoch_match else None,
            },
        )
    except Exception as error:
        task.fail(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
