import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .comet_utils import build_experiment, log_confusion_matrix, log_error_table
from .data import TextClassificationDataset, prepare_datasets
from .metrics import compute_metrics
from .model import TransformerWithHead
from .utils import get_device, set_seed


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, criterion, device, label_encoder, use_fp16: bool = False):
    """Прогон по loader без градиентов. Возвращает средний loss, полный набор метрик
    и per-sample инфо (loss/pred/proba на каждом примере) -- нужно для анализа ошибок."""
    model.eval()
    all_losses, all_preds, all_probs, all_labels, all_idx = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            idx = batch["idx"]
            batch_d = _move_to_device(batch, device)

            with torch.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(batch_d)
                losses = criterion(logits, labels)

            probs = torch.softmax(logits.float(), dim=-1)
            preds = probs.argmax(dim=-1)

            all_losses.append(losses.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_idx.append(idx if torch.is_tensor(idx) else torch.as_tensor(idx))

    all_losses = torch.cat(all_losses).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_idx = torch.cat(all_idx).numpy()

    avg_loss = float(all_losses.mean())
    metrics = compute_metrics(all_labels, all_preds, all_probs, list(label_encoder.classes_))

    per_sample = {
        "idx": all_idx,
        "y_true": all_labels,
        "y_pred": all_preds,
        "loss": all_losses,
        "probs": all_probs,
    }
    return avg_loss, metrics, per_sample


def build_error_dataframe(source_df, per_sample, label_encoder):
    df = source_df.iloc[per_sample["idx"]].copy().reset_index(drop=True)
    df["true_label"] = label_encoder.inverse_transform(per_sample["y_true"])
    df["pred_label"] = label_encoder.inverse_transform(per_sample["y_pred"])
    df["loss"] = per_sample["loss"]
    df["confidence"] = per_sample["probs"].max(axis=1)
    df["correct"] = df["true_label"] == df["pred_label"]
    return df


def run_experiment(cfg: dict, run_cfg: dict, label_encoder=None):
    """Один полный запуск: данные -> модель -> обучение с early stopping ->
    финальная оценка на test -> метрики + confusion matrix + top/bottom loss в Comet и на диск.

    run_cfg ожидает ключи:
        checkpoint, display_name, freeze_backbone, lr, head_lr,
        batch_size, eval_batch_size(optional), epochs, tags(optional)
    """
    set_seed(cfg["data"].get("seed", 42))
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(run_cfg["checkpoint"])
    train_df, val_df, test_df, label_encoder = prepare_datasets(cfg, label_encoder)
    num_labels = len(label_encoder.classes_)

    text_col = cfg["data"]["text_column"]
    max_len = cfg["data"]["max_length"]

    train_ds = TextClassificationDataset(train_df, tokenizer, text_col, max_len)
    val_ds = TextClassificationDataset(val_df, tokenizer, text_col, max_len)
    test_ds = TextClassificationDataset(test_df, tokenizer, text_col, max_len)

    bs = run_cfg.get("batch_size", cfg["training"]["batch_size"])
    ebs = run_cfg.get("eval_batch_size", cfg["training"].get("eval_batch_size", bs * 2))
    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=ebs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=ebs, shuffle=False, num_workers=num_workers)

    freeze_backbone = run_cfg.get("freeze_backbone", False)
    model = TransformerWithHead(
        checkpoint=run_cfg["checkpoint"],
        num_labels=num_labels,
        dropout=cfg["head"].get("dropout", 0.1),
        hidden_dim=cfg["head"].get("hidden_dim"),
        freeze_backbone=freeze_backbone,
    ).to(device)

    # float(...) на случай, если PyYAML распарсил scientific notation (например "2e-5") как строку
    # -- это известная особенность PyYAML при отсутствии десятичной точки в экспоненциальной записи.
    lr_backbone = float(run_cfg.get("lr", cfg["training"]["lr"]))
    lr_head = float(run_cfg.get("head_lr", cfg["training"].get("head_lr", lr_backbone)))
    wd = float(cfg["training"].get("weight_decay", 0.01))

    param_groups = model.trainable_parameter_groups(lr_backbone, lr_head, wd)
    optimizer = torch.optim.AdamW(param_groups)

    epochs = int(run_cfg.get("epochs", cfg["training"]["epochs"]))
    total_steps = max(1, len(train_loader) * epochs)
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.06))
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(reduction="none")
    use_fp16 = cfg["training"].get("fp16", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))

    mode_tag = "frozen" if freeze_backbone else "finetune"
    run_name = f"{run_cfg['display_name']}_{mode_tag}"
    experiment = build_experiment(cfg, name=run_name, tags=run_cfg.get("tags", []))
    experiment.log_parameters(
        {
            "checkpoint": run_cfg["checkpoint"],
            "freeze_backbone": freeze_backbone,
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "batch_size": bs,
            "epochs": epochs,
            "max_length": max_len,
            "weight_decay": wd,
            "warmup_ratio": cfg["training"].get("warmup_ratio", 0.06),
            "optimizer": "AdamW",
        }
    )

    best_val_f1 = -1.0
    best_state = None
    patience = cfg["training"].get("early_stopping_patience", 2)
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            batch_d = _move_to_device(batch, device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_fp16):
                logits = model(batch_d)
                loss = criterion(logits, labels).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step = epoch * len(train_loader) + step
            if step % 50 == 0:
                experiment.log_metric("train_loss_step", loss.item(), step=global_step)

        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, label_encoder, use_fp16)

        experiment.log_metric("train_loss_epoch", train_loss, epoch=epoch)
        experiment.log_metric("val_loss_epoch", val_loss, epoch=epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                experiment.log_metric(f"val_{k}", v, epoch=epoch)

        print(
            f"[{run_cfg['display_name']} | {mode_tag}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} time={time.time() - t0:.1f}s"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping на эпохе {epoch + 1} (нет улучшения val_f1_macro {patience} эпох)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, per_sample = evaluate(model, test_loader, criterion, device, label_encoder, use_fp16)
    experiment.log_metric("test_loss", test_loss)
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            experiment.log_metric(f"test_{k}", v)

    log_confusion_matrix(experiment, per_sample["y_true"], per_sample["y_pred"], list(label_encoder.classes_))

    error_df = build_error_dataframe(test_df, per_sample, label_encoder)
    out_dir = os.path.join(cfg["output"]["dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    error_df.to_csv(os.path.join(out_dir, "per_sample_errors.csv"), index=False)

    top_k = cfg["output"].get("top_k_errors", 20)
    worst = error_df.sort_values("loss", ascending=False).head(top_k)
    best = error_df.sort_values("loss", ascending=True).head(top_k)
    worst.to_csv(os.path.join(out_dir, "top_worst_losses.csv"), index=False)
    best.to_csv(os.path.join(out_dir, "top_best_losses.csv"), index=False)
    log_error_table(experiment, worst, "top_worst_losses")
    log_error_table(experiment, best, "top_best_losses")

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    experiment.log_asset(metrics_path)
    experiment.end()

    return {
        "display_name": run_cfg["display_name"],
        "checkpoint": run_cfg["checkpoint"],
        "freeze_backbone": freeze_backbone,
        "lr": lr_backbone,
        "head_lr": lr_head,
        "batch_size": bs,
        "val_f1_macro": best_val_f1,
        "test_metrics": test_metrics,
        "out_dir": out_dir,
    }
