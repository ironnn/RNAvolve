#!/usr/bin/env python
# encoding: utf-8
"""
RNAvolve (Caduceus/Mamba-RCPS backbone) full fine-tuning on the pure_te_bench TE dataset.

This reuses the public RNAvolve SFT implementation — same tokenizer, same
CDS-anchored padding, same MambaRCHeadStage2 (backbone + RC-CNN head + regressor),
same per-epoch val/test all_gather -> sklearn r2 -> CSV dump convention. We only
swap the data source to pure_te_bench so it lines up with the other train_te_*.py.

Region  : whole mRNA, CDS-anchored padding (start codon pinned at pos FIXED_CDS_START=1000)
MaxLen   : 10000 nt (TOTAL_LENGTH; RNAvolveTokenizer single-nt)
Env      : mamballm

Alignment matches finetune_stage2.py exactly:
  - RNAvolveTokenizer(model_max_length=10000), PAD=4 / N=11
  - align_and_pad_sequence: left-pad with N so the start codon sits at pos 1000,
    then pad/truncate to 10000 nt
  - backbone = stage1 weights (assets/checkpoints/backbone/pretrained_step13500.pt)

Data (same split as the other backbones):
  Train: human_train_wt + mouse_train_wt (20350)   Val: human_val_wt   Test: human_test_wt
  label = mean_te (regression), loss = MSELoss

pure_te_bench has no mutants (all WT), so dynamic sampling is off: a static
training DataLoader shuffles the complete WT training set each epoch.

Run (single GPU; backbone head batchnorm + small model -> 1 GPU is simplest):
  conda run -n mamballm python external_models/training/train_te_rnavolve.py \
      --epochs 10 --bsz 12 --grad_accum 4 --devices 1 --output_dir ./te_rnavolve_out
Multi-GPU DDP:
  python external_models/training/train_te_rnavolve.py \
      --epochs 10 --devices 7 --output_dir ./te_rnavolve_out
"""
import os
import sys
import argparse

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

# --- RNAvolve repo paths ---
RNAVOLVE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RNAVOLVE_ROOT)
sys.path.insert(0, os.path.join(RNAVOLVE_ROOT, "src"))

from rnavolve.models.tokenization import RNAvolveTokenizer
from rnavolve.training.train_student import (
    RNAPaddedSeqDataset, collate_fn, build_backbone, RNAvolveSFTModule,
)

# Padding constants — identical to finetune_stage2.py.
TOTAL_LENGTH = 10000
FIXED_CDS_START = 1000


def align_and_pad_sequence(sequence, utr5_length):
    """CDS-anchored padding: pin the start codon at FIXED_CDS_START, pad/clip to TOTAL_LENGTH.
    Verbatim from finetune_stage2.py."""
    sequence = str(sequence)
    left_padding = max(FIXED_CDS_START - int(utr5_length), 0)
    new_sequence = "N" * left_padding + sequence
    new_sequence = new_sequence[:TOTAL_LENGTH]
    new_sequence = new_sequence.ljust(TOTAL_LENGTH, "N")
    return new_sequence


def load_split(path):
    """Read a pure_te_bench parquet, CDS-align+pad, return a df with the columns
    MambaRCHeadStage2 / RNAPaddedSeqDataset expect (human_seq_pad, final_label)."""
    import pyarrow.parquet as pq
    df = pq.read_table(path, columns=["mrna", "utr5_size", "mean_te"]).to_pandas()
    df["human_seq_pad"] = df.apply(
        lambda r: align_and_pad_sequence(r["mrna"], r["utr5_size"]), axis=1)
    df["final_label"] = df["mean_te"].astype("float32")
    return df[["human_seq_pad", "final_label"]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="assets/training_data/common_te_splits")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=4)
    # Unified update rate: backbone + head share ONE lr, matching the other
    # train_te_*.py full-FT scripts (single --lr, default 1e-5). The shared
    # MambaRCHeadStage2.configure_optimizers keeps its two param groups, but we
    # feed them the same value via head_lr == backbone_lr == args.lr.
    # To reproduce the ORIGINAL finetune_stage2 differential-lr scheme, pass
    # --head_lr 1e-3 --backbone_lr 5e-6 (these override --lr when given).
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=None,
                    help="override head lr (default: --lr). Original stage2 used 1e-3.")
    ap.add_argument("--backbone_lr", type=float, default=None,
                    help="override backbone lr (default: --lr). Original stage2 used 5e-6.")
    ap.add_argument("--devices", type=int, default=1, help="number of GPUs (DDP if >1)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default="./te_rnavolve_out")
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--max_train", type=int, default=-1)
    args = ap.parse_args()

    # head/backbone lr default to the unified --lr; override individually if given.
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[TE-rnavolve] region=mRNA(CDS-anchored@{FIXED_CDS_START}) "
          f"max_len={TOTAL_LENGTH}nt env=mamballm devices={args.devices} "
          f"head_lr={head_lr} backbone_lr={backbone_lr}", flush=True)

    # --- tokenizer (same as stage2) ---
    tokenizer = RNAvolveTokenizer(model_max_length=TOTAL_LENGTH)

    # --- data: train = human + mouse, val/test = human ---
    d = args.data_dir
    tr_df = pd.concat([
        load_split(os.path.join(d, "human_train_wt.parquet")),
        load_split(os.path.join(d, "mouse_train_wt.parquet")),
    ], ignore_index=True)
    val_df = load_split(os.path.join(d, "human_val_wt.parquet"))
    test_df = load_split(os.path.join(d, "human_test_wt.parquet"))
    if args.max_train > 0:
        tr_df = tr_df.head(args.max_train)
        val_df = val_df.head(64); test_df = test_df.head(64)
    print(f"[data] train={len(tr_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    train_loader = DataLoader(
        RNAPaddedSeqDataset(tr_df["human_seq_pad"].tolist(), tr_df["final_label"].tolist(), tokenizer),
        batch_size=args.bsz, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    val_loader = DataLoader(
        RNAPaddedSeqDataset(val_df["human_seq_pad"].tolist(), val_df["final_label"].tolist(), tokenizer),
        batch_size=args.bsz, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(
        RNAPaddedSeqDataset(test_df["human_seq_pad"].tolist(), test_df["final_label"].tolist(), tokenizer),
        batch_size=args.bsz, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # --- backbone: stage1 weights, exactly as finetune_stage2.py ---
    config_path = os.path.join(RNAVOLVE_ROOT, "backbone", "configs", "mamba_motif_moe.yaml")
    stage1_weights = os.path.join(RNAVOLVE_ROOT, "assets", "checkpoints", "backbone", "pretrained_step13500.pt")
    stage1_backbone = build_backbone(config_path, stage1_weights)

    model = RNAvolveSFTModule(
        backbone_model=stage1_backbone,
        head_name="mambarc",
        tokenizer=tokenizer,
        save_dir=args.output_dir,
        dynamic_sampling=False,
        batch_size=args.bsz,
        num_workers=args.num_workers,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
    )

    ckpt = ModelCheckpoint(
        dirpath=args.output_dir, filename="best-{epoch:02d}-{val_r2_global:.4f}",
        monitor="val_r2_global", mode="max", save_top_k=1, save_weights_only=True)
    csv_logger = CSVLogger(save_dir=args.output_dir, name="", version="")

    trainer = Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision="bf16-mixed", log_every_n_steps=5,
        callbacks=[ckpt], logger=csv_logger,
        accumulate_grad_batches=args.grad_accum,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader])
    if trainer.is_global_zero:
        print(f"BEST val_r2_global = {ckpt.best_model_score} @ {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
