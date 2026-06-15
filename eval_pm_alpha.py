import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ALPHA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class PMSample:
    # 单个 PM 样本同时保留相对路径和实际读取路径，方便后续输出表格和读图。
    sample_id: str
    image_rel: str
    alpha_rel: str
    image_path: Path
    alpha_path: Path


def parse_args():
    """解析命令行参数，告诉脚本去哪里找测试集、预测结果、prompt 和输出目录。"""
    parser = argparse.ArgumentParser(
        description="Evaluate alpha predictions and save PM visualization figures."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--image-dir", type=str, default="image_resize")
    parser.add_argument("--alpha-dir", type=str, default="alpha_resize")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--pred-pattern", type=str, default="{id}.png")
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--prompt-pattern", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-resize-pred",
        action="store_false",
        dest="resize_pred",
        help="Fail if prediction and GT alpha sizes differ.",
    )
    parser.set_defaults(resize_pred=True)
    return parser.parse_args()


def main():
    """串起完整评估流程：读 split、找 prompt、找预测、算指标、存结果。"""
    args = parse_args()
    samples = load_split(args.split, args.dataset_root, args.image_dir, args.alpha_dir)
    vis_dir = args.vis_dir
    if args.save_visualizations and vis_dir is None:
        vis_dir = args.output_dir / "vis"
    prompt_dir = args.prompt_dir
    if prompt_dir is None:
        parent_prompt_dir = args.pred_dir.parent / "prompt"
        child_prompt_dir = args.pred_dir / "prompt"
        prompt_dir = parent_prompt_dir if parent_prompt_dir.exists() else child_prompt_dir

    rows, summary = evaluate_predictions(
        samples,
        args.pred_dir,
        pred_pattern=args.pred_pattern,
        prompt_dir=prompt_dir,
        prompt_pattern=args.prompt_pattern,
        resize_pred=args.resize_pred,
        vis_dir=vis_dir,
        limit=args.limit,
    )
    write_metric_outputs(
        rows,
        summary,
        args.output_dir,
        extra={
            "dataset_root": str(args.dataset_root),
            "split": str(args.split),
            "pred_dir": str(args.pred_dir),
            "pred_pattern": args.pred_pattern,
            "prompt_dir": str(prompt_dir),
            "prompt_pattern": args.prompt_pattern,
            "resize_pred": args.resize_pred,
            "vis_dir": str(vis_dir) if vis_dir else None,
        },
    )
    print(
        f"Evaluated {summary['count']} samples: "
        f"MAE={summary['mae']:.6f}, MSE={summary['mse']:.6f}, SAD={summary['sad']:.6f}"
    )
    print(f"Wrote metrics to {args.output_dir}")
    if vis_dir is not None:
        print(f"Wrote visualizations to {vis_dir}")


def natural_key(value: str) -> List[object]:
    """排序字符串中的数字部分，比如 image_2 排在 image_10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_path(path, root: Path) -> Path:
    """把相对路径补成绝对路径；如果本来就是绝对路径则直接返回。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def sample_id_from_relpath(path: str) -> str:
    """把相对文件路径转成稳定的样本编号，便于写表和命名输出文件。"""
    return Path(path).with_suffix("").as_posix().replace("/", "__")


def load_split(
    split_path: Path,
    dataset_root: Path,
    image_dir: str = "image_resize",
    alpha_dir: str = "alpha_resize",
) -> List[PMSample]:
    """读取 split 文件，整理成统一的样本列表。"""
    dataset_root = dataset_root.resolve()
    image_root = resolve_path(image_dir, dataset_root)
    alpha_root = resolve_path(alpha_dir, dataset_root)
    samples = []

    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t, ]+", line)
        if len(parts) == 1:
            stem = parts[0]
            image_path = resolve_stem(image_root, stem, IMAGE_EXTENSIONS)
            alpha_path = resolve_stem(alpha_root, stem, ALPHA_EXTENSIONS)
            image_rel = image_path.relative_to(dataset_root).as_posix()
            alpha_rel = alpha_path.relative_to(dataset_root).as_posix()
        else:
            image_rel, alpha_rel = parts[0], parts[1]
            image_path = resolve_path(image_rel, dataset_root)
            alpha_path = resolve_path(alpha_rel, dataset_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for split entry: {image_path}")
        if not alpha_path.exists():
            raise FileNotFoundError(f"Alpha not found for split entry: {alpha_path}")
        samples.append(
            PMSample(
                sample_id=sample_id_from_relpath(image_rel),
                image_rel=image_rel,
                alpha_rel=alpha_rel,
                image_path=image_path,
                alpha_path=alpha_path,
            )
        )

    if not samples:
        raise ValueError(f"No samples found in split file: {split_path}")
    return samples


def resolve_stem(root: Path, stem: str, extensions: Sequence[str]) -> Path:
    """在给定目录下按常见扩展名查找同名文件。"""
    for ext in extensions:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No file for stem '{stem}' under {root}")


def load_rgb(path: Path) -> np.ndarray:
    """读取 RGB 图像，供可视化拼图使用。"""
    return np.asarray(Image.open(path).convert("RGB"))


def load_alpha(path: Path) -> np.ndarray:
    """读取 alpha 并归一化到 [0, 1]，方便统一计算指标。"""
    alpha = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    return np.clip(alpha / 255.0, 0.0, 1.0)


def load_prompt(path: Path) -> np.ndarray:
    """读取 prompt 图，并统一转成 RGB，方便和其他小图拼接。"""
    return np.asarray(Image.open(path).convert("RGB"))


def find_prediction(pred_dir: Path, sample: PMSample, pred_pattern: str) -> Path:
    """根据样本信息在预测目录里找到对应的 alpha 文件。"""
    stem = Path(sample.image_rel).stem
    image_name = Path(sample.image_rel).name
    alpha_name = Path(sample.alpha_rel).name
    candidates = [
        pred_dir
        / pred_pattern.format(
            id=sample.sample_id,
            stem=stem,
            image_name=image_name,
            alpha_name=alpha_name,
        ),
        pred_dir / f"{sample.sample_id}.png",
        pred_dir / f"{stem}.png",
        pred_dir / alpha_name,
        pred_dir / Path(sample.alpha_rel),
    ]
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        if candidate.exists():
            return candidate
        unique_candidates.append(candidate)
        seen.add(candidate)
    raise FileNotFoundError(
        "Prediction not found for "
        f"{sample.sample_id}. Tried: {', '.join(str(path) for path in unique_candidates)}"
    )


def find_prompt(prompt_dir: Path, sample: PMSample, prompt_pattern: Optional[str]) -> Path:
    """根据样本信息在 prompt 目录里找到对应的提示图。"""
    stem = Path(sample.image_rel).stem
    image_name = Path(sample.image_rel).name
    alpha_name = Path(sample.alpha_rel).name
    candidates = []
    if prompt_pattern:
        candidates.append(
            prompt_dir
            / prompt_pattern.format(
                id=sample.sample_id,
                stem=stem,
                image_name=image_name,
                alpha_name=alpha_name,
            )
        )
    candidates.extend(
        [
            prompt_dir / f"{sample.sample_id}.png",
            prompt_dir / f"{sample.sample_id}.jpg",
            prompt_dir / f"{stem}.png",
            prompt_dir / f"{stem}.jpg",
            prompt_dir / image_name,
            prompt_dir / alpha_name,
        ]
    )
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        if candidate.exists():
            return candidate
        unique_candidates.append(candidate)
        seen.add(candidate)
    raise FileNotFoundError(
        "Prompt not found for "
        f"{sample.sample_id}. Tried: {', '.join(str(path) for path in unique_candidates)}"
    )


def evaluate_predictions(
    samples: Sequence[PMSample],
    pred_dir: Path,
    *,
    pred_pattern: str,
    prompt_dir: Path,
    prompt_pattern: Optional[str],
    resize_pred: bool,
    vis_dir: Optional[Path],
    limit: Optional[int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """对整批样本逐张评估，并可选保存每张图的对比可视化。"""
    rows = []
    selected_samples = samples[:limit] if limit is not None else samples
    for sample in selected_samples:
        pred_path = find_prediction(pred_dir, sample, pred_pattern)
        prompt_path = find_prompt(prompt_dir, sample, prompt_pattern)
        true_alpha = load_alpha(sample.alpha_path)
        pred_alpha = load_alpha(pred_path)
        prompt_rgb = pad_prompt_to_shape(load_prompt(prompt_path), true_alpha.shape)
        if pred_alpha.shape != true_alpha.shape:
            if not resize_pred:
                raise ValueError(
                    f"Shape mismatch for {sample.sample_id}: "
                    f"pred={pred_alpha.shape}, true={true_alpha.shape}"
                )
            pred_alpha = resize_alpha(pred_alpha, true_alpha.shape, Image.Resampling.BILINEAR)
        metrics = compute_alpha_metrics(pred_alpha, true_alpha)
        rows.append(
            {
                "id": sample.sample_id,
                "image": sample.image_rel,
                "alpha": sample.alpha_rel,
                "prompt": str(prompt_path),
                "prediction": str(pred_path),
                **metrics,
            }
        )
        if vis_dir is not None:
            make_comparison_figure(
                load_rgb(sample.image_path),
                prompt_rgb,
                true_alpha,
                pred_alpha,
                vis_dir / f"{sample.sample_id}.png",
                title=sample.sample_id,
            )
    return rows, summarize_metric_rows(rows)


def compute_alpha_metrics(pred_alpha: np.ndarray, true_alpha: np.ndarray) -> Dict[str, float]:
    """计算单张图的 MAE、MSE、SAD。"""
    pred_alpha = np.clip(pred_alpha.astype(np.float32), 0.0, 1.0)
    true_alpha = np.clip(true_alpha.astype(np.float32), 0.0, 1.0)
    diff = pred_alpha - true_alpha
    abs_diff = np.abs(diff)
    return {
        "mae": float(abs_diff.mean()),
        "mse": float(np.square(diff).mean()),
        "sad": float(abs_diff.sum() / 1000.0),
    }


def summarize_metric_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """把逐图指标汇总成整套测试集的平均结果。"""
    if not rows:
        raise ValueError("Cannot summarize an empty metric table.")
    summary = {"count": len(rows)}
    for name in ["mae", "mse", "sad"]:
        summary[name] = float(np.mean([float(row[name]) for row in rows]))
    return summary


def write_metric_outputs(
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    output_dir: Path,
    *,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """把指标结果写成 JSON 和 CSV，便于提交、复查和绘图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = dict(summary)
    if extra:
        summary_payload["config"] = extra
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = ["id", "image", "alpha", "prompt", "prediction", "mae", "mse", "sad"]
    with (output_dir / "metrics_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_comparison_figure(
    image_rgb: np.ndarray,
    prompt_rgb: np.ndarray,
    true_alpha: np.ndarray,
    pred_alpha: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    """生成 2x3 的可视化图"""
    if image_rgb.shape[:2] != true_alpha.shape:
        image_rgb = resize_rgb(image_rgb, true_alpha.shape, Image.Resampling.BOX)
    if prompt_rgb.shape[:2] != true_alpha.shape:
        prompt_rgb = resize_rgb(prompt_rgb, true_alpha.shape, Image.Resampling.NEAREST)
    true_tile = gray_to_rgb(true_alpha)
    pred_tile = gray_to_rgb(pred_alpha)
    error_tile = colorize_error(np.abs(pred_alpha - true_alpha))
    cutout_tile = composite_on_checker(image_rgb, pred_alpha)
    top_row = np.concatenate(
        [
            with_label(image_rgb, "Image"),
            with_label(prompt_rgb, "Prompt"),
            with_label(true_tile, "GT alpha"),
        ],
        axis=1,
    )
    bottom_row = np.concatenate(
        [
            with_label(pred_tile, "Pred alpha"),
            with_label(error_tile, "Abs error"),
            with_label(cutout_tile, "Pred cutout"),
        ],
        axis=1,
    )
    canvas = np.concatenate([top_row, bottom_row], axis=0)
    if title:
        title_bar = make_text_bar(canvas.shape[1], title)
        canvas = np.concatenate([title_bar, canvas], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)


def gray_to_rgb(alpha: np.ndarray) -> np.ndarray:
    """把单通道 alpha 转成三通道灰度图，方便和其他彩色图拼接"""
    alpha_u8 = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(alpha_u8[..., None], 3, axis=2)


def resize_rgb(image_rgb: np.ndarray, target_shape: Tuple[int, int], resample) -> np.ndarray:
    """把 RGB 图缩放到指定高宽。"""
    target_h, target_w = target_shape
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((target_w, target_h), resample=resample))


def resize_alpha(alpha: np.ndarray, target_shape: Tuple[int, int], resample) -> np.ndarray:
    """把 alpha 图缩放到指定高宽，并继续保持 [0, 1] 范围。"""
    target_h, target_w = target_shape
    alpha_u8 = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(alpha_u8, mode="L")
    resized = np.asarray(image.resize((target_w, target_h), resample=resample)).astype(np.float32)
    return np.clip(resized / 255.0, 0.0, 1.0)


def colorize_error(error: np.ndarray) -> np.ndarray:
    """把绝对误差转成近似 inferno 风格的热力图。"""
    anchors_x = np.array([0, 64, 128, 192, 255], dtype=np.float32)
    anchors_rgb = np.array(
        [
            [0, 0, 4],
            [87, 15, 109],
            [187, 55, 84],
            [249, 142, 8],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    error_u8 = np.round(np.clip(error, 0.0, 1.0) * 255.0).astype(np.float32)
    channels = [np.interp(error_u8, anchors_x, anchors_rgb[:, i]) for i in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def pad_prompt_to_shape(prompt_rgb: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """把较小的 prompt 图用白色居中补到目标尺寸，保持和最终可视化一致。"""
    target_h, target_w = target_shape
    h, w = prompt_rgb.shape[:2]
    if (h, w) == (target_h, target_w):
        return prompt_rgb
    if h > target_h or w > target_w:
        raise ValueError(f"Prompt larger than target alpha: prompt={prompt_rgb.shape}, target={target_shape}")
    canvas = np.full((target_h, target_w, 3), 255, dtype=prompt_rgb.dtype)
    top = (target_h - h) // 2
    left = (target_w - w) // 2
    canvas[top : top + h, left : left + w] = prompt_rgb
    return canvas


def with_label(tile: np.ndarray, label: str) -> np.ndarray:
    """在每个小图上方加一条标签栏，说明这一列是什么内容。"""
    label_bar = make_text_bar(tile.shape[1], label)
    return np.concatenate([label_bar, tile], axis=0)


def make_text_bar(width: int, text: str) -> np.ndarray:
    """生成小图顶部的文字标签栏。"""
    image = Image.new("RGB", (width, 28), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), text, fill=(30, 30, 30), font=get_label_font())
    return np.asarray(image)


def get_label_font():
    """加载标签字体；找不到系统字体时使用 PIL 默认字体。"""
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def composite_on_checker(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """把预测 alpha 叠加到棋盘格背景上，便于查看抠图效果。"""
    h, w = alpha.shape
    yy, xx = np.indices((h, w))
    checker = np.where(((xx // 16) + (yy // 16)) % 2 == 0, 230, 180).astype(np.uint8)
    checker = np.repeat(checker[..., None], 3, axis=2)
    alpha_3 = alpha[..., None].astype(np.float32)
    return np.round(image_rgb.astype(np.float32) * alpha_3 + checker * (1.0 - alpha_3)).astype(
        np.uint8
    )


if __name__ == "__main__":
    main()
