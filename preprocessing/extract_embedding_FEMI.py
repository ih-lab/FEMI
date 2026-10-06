"""
Adapted from extract_embeddings.py from Shenni
---------------------
Extracts and saves FEMI [CLS] embeddings for the cropped images written by
`eval_segmenter.py --mode pipeline --save-crops`, then compares the
embeddings of the same image across pipeline configs. Inference only: the
pretrained FEMI encoder is never trained or updated.

Expected input layout (eval_segmenter.py's crops/ folder):
    {crops_dir}/{config}/{dataset}/{image}

Run from preprocessing/ with the FEMI environment:
    python extract_embedding_FEMI.py
    python extract_embedding_FEMI.py --crops_dir segmenter_pipeline_eval_output/new_valid/crops
    python extract_embedding_FEMI.py --configs A,C --overwrite

Outputs:
    {output_dir}/{config}/embeddings.npy   shape (N, 1024)
    {output_dir}/{config}/index.csv         N rows: config, dataset, image, filepath
    {output_dir}/pairwise_similarity.csv    one row per (image, config pair)
    {output_dir}/summary.csv                one row per config pair
    {output_dir}/femi_similarity_summary.png   mean +/- std cosine per config pair,
                                               with every image as a dot
    {output_dir}/femi_similarity_heatmap.png   per-image cosine, images x config pairs
    {output_dir}/femi_embedding_pca.png        2D PCA of all embeddings, colored by
                                               config, same image joined by a line

Resumable: if embeddings.npy already exists for a config, extraction is
skipped for it unless --overwrite is passed (the comparison and plots are
always rerun, and FEMI is only loaded if something needs extracting).
Plots need matplotlib; without it the CSVs are still written.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CROPS_DIR = SCRIPT_DIR / "segmenter_crops_output" / "crops"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "segmenter_crops_output" / "femi_embeddings"
DEFAULT_FEMI_CKPT = SCRIPT_DIR.parent / "torch_model_maskRatio075_18mill"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    """Loads frames from a list of filepaths and applies preprocessing."""

    def __init__(self, filepaths: list[str], preprocess):
        self.filepaths  = filepaths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img)
        except Exception as e:
            log.warning(f"Failed to load {path}: {e} — returning zeros")
            return torch.zeros(3, 224, 224)


def discover_crops(crops_dir: Path, configs: list[str] | None) -> pd.DataFrame:
    """One row per image under {crops_dir}/{config}/{dataset}/."""
    rows = []
    for config_dir in sorted(p for p in crops_dir.iterdir() if p.is_dir()):
        if configs and config_dir.name not in configs:
            continue
        for dataset_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            for img_path in sorted(dataset_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    rows.append({
                        "config": config_dir.name,
                        "dataset": dataset_dir.name,
                        "image": img_path.name,
                        "filepath": str(img_path),
                    })
    return pd.DataFrame(rows, columns=["config", "dataset", "image", "filepath"])


# ── Model ─────────────────────────────────────────────────────────────────

class FEMIEncoder:
    """FEMI's ViT-MAE encoder with the same input preprocessing as
    main_pretrain.py (224x224, [0, 1], ImageNet normalization)."""

    embed_dim = 1024
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from transformers import ViTMAEModel

        self.device = device
        self.model = ViTMAEModel.from_pretrained(checkpoint_path, mask_ratio=0.0).to(device).eval()
        self.num_patches = self.model.embeddings.patch_embeddings.num_patches

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr.transpose(2, 0, 1))

    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device)
        # mask_ratio=0 keeps every patch; a fixed noise keeps patch order deterministic
        noise = torch.arange(self.num_patches, dtype=torch.float32, device=self.device)
        noise = noise.expand(batch.shape[0], -1)
        hidden = self.model(pixel_values=batch, noise=noise).last_hidden_state
        return hidden[:, 0].cpu().numpy()


# ── Extraction ────────────────────────────────────────────────────────────

def extract(
    model: FEMIEncoder,
    config: str,
    meta: pd.DataFrame,
    output_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    out_dir = Path(output_dir) / config
    emb_path = out_dir / "embeddings.npy"
    idx_path = out_dir / "index.csv"

    if emb_path.exists() and not overwrite:
        log.info(f"[{config}] embeddings already exist at {emb_path} — skipping.")
        log.info("Pass --overwrite to re-extract.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{config}] {len(meta)} frames to embed")

    dataset = FrameDataset(meta["filepath"].tolist(), model.preprocess)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_embeddings = []
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        emb = model.get_embeddings(batch)   # (B, D)
        all_embeddings.append(emb)
        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            log.info(f"[{config}] batch {i+1}/{n_batches}")

    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, D)
    assert embeddings.shape == (len(meta), model.embed_dim), (
        f"Shape mismatch: got {embeddings.shape}, "
        f"expected ({len(meta)}, {model.embed_dim})"
    )

    np.save(emb_path, embeddings)
    meta.to_csv(idx_path, index=False)

    log.info(f"[{config}] saved embeddings {embeddings.shape} → {emb_path}")
    log.info(f"[{config}] saved index ({len(meta)} rows) → {idx_path}")


# ── Comparison ────────────────────────────────────────────────────────────

def load_config_embeddings(output_dir: str, config: str) -> pd.DataFrame:
    """Saved embeddings for one config, keyed by (dataset, image)."""
    out_dir = Path(output_dir) / config
    emb = np.load(out_dir / "embeddings.npy")
    idx = pd.read_csv(out_dir / "index.csv")
    idx["embedding"] = list(emb)
    return idx.set_index(["dataset", "image"])["embedding"]


def compare_configs(output_dir: str, configs: list[str]) -> None:
    """Cosine similarity / L2 distance of the same image across every pair
    of configs. 1.0 cosine means FEMI produced the same embedding."""
    if len(configs) < 2:
        log.info("Only one config - nothing to compare.")
        return

    per_config = {c: load_config_embeddings(output_dir, c) for c in configs}
    pair_rows, summary_rows = [], []

    for a, b in itertools.combinations(configs, 2):
        shared = per_config[a].index.intersection(per_config[b].index)
        if len(shared) == 0:
            log.warning(f"[{a}-{b}] no images in common - skipping.")
            continue
        ea = np.stack(per_config[a].loc[shared].to_numpy())
        eb = np.stack(per_config[b].loc[shared].to_numpy())
        cos = np.sum(ea * eb, axis=1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1))
        l2 = np.linalg.norm(ea - eb, axis=1)

        for (dataset, image), c, d in zip(shared, cos, l2):
            pair_rows.append({"dataset": dataset, "image": image, "pair": f"{a}-{b}",
                              "cls_cosine": round(float(c), 6), "cls_l2": round(float(d), 4)})
        worst = shared[int(np.argmin(cos))]
        summary_rows.append({
            "pair": f"{a}-{b}", "n_images": len(shared),
            "cls_cosine_mean": round(float(cos.mean()), 6),
            "cls_cosine_std": round(float(cos.std()), 6),
            "cls_cosine_min": round(float(cos.min()), 6),
            "cls_l2_mean": round(float(l2.mean()), 4),
            "least_similar_image": f"{worst[0]}/{worst[1]}",
        })

    if not summary_rows:
        return
    pd.DataFrame(pair_rows).to_csv(Path(output_dir) / "pairwise_similarity.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(Path(output_dir) / "summary.csv", index=False)

    log.info("CLS-embedding similarity between configs (1.0 = identical):")
    for r in summary_rows:
        log.info(f"  {r['pair']}: n={r['n_images']} cosine mean={r['cls_cosine_mean']:.4f} "
                 f"std={r['cls_cosine_std']:.4f} min={r['cls_cosine_min']:.4f} "
                 f"(least similar: {r['least_similar_image']})")
    log.info(f"Comparison written to {Path(output_dir) / 'summary.csv'} "
             f"and {Path(output_dir) / 'pairwise_similarity.csv'}")


# ── Plots ─────────────────────────────────────────────────────────────────

def _image_label(dataset: str, image: str, n_datasets: int) -> str:
    stem = os.path.splitext(image)[0]
    return f"{dataset}/{stem}" if n_datasets > 1 else stem


def plot_similarity_summary(pairs: pd.DataFrame, out_path: Path, plt) -> None:
    """Distribution of cosine similarity per config pair. Individual images
    are drawn as dots only when there are few enough to read; otherwise a
    box plot, since thousands of dots just paint over the bars."""
    order = list(dict.fromkeys(pairs["pair"]))
    grouped = [pairs.loc[pairs["pair"] == pair, "cls_cosine"].to_numpy() for pair in order]
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(order)), 4.5))
    if max(len(v) for v in grouped) <= 80:
        rng = np.random.default_rng(0)
        for i, vals in enumerate(grouped):
            ax.bar(i, vals.mean(), yerr=vals.std(), width=0.6, color="#9ecae1",
                   edgecolor="#3182bd", capsize=6, zorder=1)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=22,
                       color="#08519c", zorder=2)
    else:
        ax.boxplot(grouped, positions=range(len(order)), widths=0.55, showfliers=True,
                   flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    lo = pairs["cls_cosine"].min()
    ax.set_ylim(max(0.0, lo - 0.02), 1.002)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Pipeline pair")
    ax.set_ylabel("Cosine similarity of FEMI [CLS] embedding")
    ax.set_title("Same image, different crop pipeline\n(1.0 = FEMI output unchanged)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


HEATMAP_MAX_ROWS = 40


def plot_similarity_heatmap(pairs: pd.DataFrame, out_path: Path, plt) -> None:
    """Images (rows) x config pairs (columns), colored by cosine similarity.
    With more images than fit on a page, only the least-similar ones are
    shown — a row per image of the full set exceeds matplotlib's size limit."""
    n_datasets = pairs["dataset"].nunique()
    pairs = pairs.assign(label=[_image_label(d, i, n_datasets)
                                for d, i in zip(pairs["dataset"], pairs["image"])])
    table = pairs.pivot(index="label", columns="pair", values="cls_cosine")
    table = table[list(dict.fromkeys(pairs["pair"]))]
    table = table.loc[table.min(axis=1).sort_values().index]
    shown_of = None
    if len(table) > HEATMAP_MAX_ROWS:
        shown_of = len(table)
        table = table.iloc[:HEATMAP_MAX_ROWS]

    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * table.shape[1], 1.2 + 0.35 * table.shape[0]))
    im = ax.imshow(table.to_numpy(), cmap="viridis", aspect="auto",
                   vmin=table.to_numpy().min(), vmax=1.0)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            v = table.iat[r, c]
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v < table.to_numpy().mean() else "black")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_xlabel("Pipeline pair")
    title = "Per-image FEMI embedding similarity\n(least similar at top)"
    if shown_of is not None:
        title += f"\n{len(table)} of {shown_of} images shown"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_pca(output_dir: str, configs: list[str], out_path: Path, plt) -> None:
    """2D PCA of every config's embeddings. Points for the same image are
    joined by a gray line, so long lines = crops that moved FEMI's output."""
    per_config = {c: load_config_embeddings(output_dir, c) for c in configs}
    shared = per_config[configs[0]].index
    for c in configs[1:]:
        shared = shared.intersection(per_config[c].index)
    if len(shared) == 0:
        log.warning("No images shared by every config - skipping PCA plot.")
        return

    stacked = np.concatenate([np.stack(per_config[c].loc[shared].to_numpy()) for c in configs])
    centered = stacked - stacked.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    explained = (s[:2] ** 2) / np.sum(s ** 2)
    coords = coords.reshape(len(configs), len(shared), 2)

    # Lines for every image are readable only for small sets. On a large set,
    # join just the images whose pipelines moved farthest apart.
    if len(shared) <= 80:
        line_idx = np.arange(len(shared))
    else:
        span = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0), axis=1)
        line_idx = np.argsort(span)[-40:]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for j in line_idx:
        ax.plot(coords[:, j, 0], coords[:, j, 1], color="lightgray", lw=1, zorder=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    point_size = 36 if len(shared) <= 80 else 8
    for k, c in enumerate(configs):
        ax.scatter(coords[k, :, 0], coords[k, :, 1], s=point_size, color=colors[k % len(colors)],
                   label=c, zorder=2, alpha=0.7, edgecolor="none")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
    title = "FEMI embeddings by crop pipeline\n(gray line joins the same image"
    title += ")" if len(shared) <= 80 else ", 40 that moved the most)"
    ax.set_title(title)
    ax.legend(title="Pipeline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(output_dir: str, configs: list[str]) -> None:
    pair_path = Path(output_dir) / "pairwise_similarity.csv"
    if len(configs) < 2 or not pair_path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed - skipping plots (CSVs were still written).")
        return

    pairs = pd.read_csv(pair_path)
    out = Path(output_dir)
    plot_similarity_summary(pairs, out / "femi_similarity_summary.png", plt)
    plot_similarity_heatmap(pairs, out / "femi_similarity_heatmap.png", plt)
    plot_embedding_pca(output_dir, configs, out / "femi_embedding_pca.png", plt)
    log.info(f"Plots written to {out}: femi_similarity_summary.png, "
             "femi_similarity_heatmap.png, femi_embedding_pca.png")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crops_dir", default=str(DEFAULT_CROPS_DIR),
        help="eval_segmenter.py crops folder: {crops_dir}/{config}/{dataset}/{image}",
    )
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated subset of configs to run, e.g. A,C (default: every folder in crops_dir)",
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory for embeddings output",
    )
    parser.add_argument(
        "--femi_ckpt", default=str(DEFAULT_FEMI_CKPT),
        help="PyTorch FEMI checkpoint folder (Hugging Face ViT-MAE format)",
    )
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    if not crops_dir.is_dir():
        sys.exit(f"ERROR: crops directory not found: {crops_dir} "
                 "(run eval_segmenter.py --mode pipeline --save-crops first)")

    wanted = [c.strip() for c in args.configs.split(",")] if args.configs else None
    meta_all = discover_crops(crops_dir, wanted)
    if meta_all.empty:
        sys.exit(f"ERROR: no cropped images found under {crops_dir}")
    configs = sorted(meta_all["config"].unique())
    log.info(f"Found configs {configs} under {crops_dir}")

    to_extract = [c for c in configs
                  if args.overwrite or not (Path(args.output_dir) / c / "embeddings.npy").exists()]
    model = None
    if to_extract:
        model = FEMIEncoder(args.femi_ckpt, device=args.device)
        log.info(f"Loaded FEMI encoder from {args.femi_ckpt} on {args.device} (inference only)")

    for config in configs:
        if config not in to_extract:
            log.info(f"[{config}] embeddings already exist — skipping (pass --overwrite to re-extract).")
            continue
        extract(
            model        = model,
            config       = config,
            meta         = meta_all[meta_all["config"] == config].reset_index(drop=True),
            output_dir   = args.output_dir,
            batch_size   = args.batch_size,
            num_workers  = args.num_workers,
            overwrite    = args.overwrite,
        )

    compare_configs(args.output_dir, configs)
    plot_comparison(args.output_dir, configs)
