import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ALPHA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class PMSample:
    # split 中只保存相对路径，后续换数据集根目录时不需要改文件内容。
    sample_id: str
    image_rel: str
    alpha_rel: str


def parse_args():
    """解析命令行参数，指定 PM 数据集位置、输出 split 路径和写入方式。"""
    parser = argparse.ArgumentParser(description="Create or update the unified PM alpha split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=str, default="image_resize")
    parser.add_argument("--alpha-dir", type=str, default="alpha_resize")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite the split.")
    parser.add_argument(
        "--append-new",
        action="store_true",
        help="Keep existing entries and append newly discovered image/alpha pairs.",
    )
    return parser.parse_args()


def main():
    """执行 split 生成流程：扫描样本、处理覆盖/追加模式、写出结果。"""
    args = parse_args()
    discovered = discover_pm_samples(args.dataset_root, args.image_dir, args.alpha_dir)
    if not discovered:
        raise RuntimeError(
            f"No matched PM samples found in {args.dataset_root}/{args.image_dir} "
            f"and {args.dataset_root}/{args.alpha_dir}"
        )

    if args.output.exists() and not args.overwrite and not args.append_new:
        print(f"Split already exists: {args.output}")
        print("Use --overwrite to recreate it or --append-new to add new matched samples.")
        return

    samples = discovered
    if args.output.exists() and args.append_new and not args.overwrite:
        existing = load_split_entries(args.output)
        existing_ids = {sample.sample_id for sample in existing}
        samples = existing + [sample for sample in discovered if sample.sample_id not in existing_ids]

    write_split(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


def natural_key(value: str) -> List[object]:
    """排序带数字的文件名，避免 10 排在 2 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_path(path: str, root: Path) -> Path:
    """把用户传入的相对路径转换到数据集根目录下。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def sample_id_from_relpath(path: str) -> str:
    """由图像相对路径生成样本编号，用于去重和后续结果命名。"""
    return Path(path).with_suffix("").as_posix().replace("/", "__")


def iter_files(root: Path, extensions: Sequence[str]) -> List[Path]:
    """递归收集指定类型的图像文件，并按自然顺序排列。"""
    extensions = tuple(ext.lower() for ext in extensions)
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions],
        key=lambda path: natural_key(path.relative_to(root).as_posix()),
    )


def index_by_relative_stem(paths: Sequence[Path], base_dir: Path) -> Dict[str, Path]:
    """用去掉扩展名后的相对文件名建立索引，便于匹配 image 和 alpha。"""
    return {path.relative_to(base_dir).with_suffix("").as_posix(): path for path in paths}


def discover_pm_samples(dataset_root: Path, image_dir: str, alpha_dir: str) -> List[PMSample]:
    """扫描图像和 alpha 文件夹，找出两边都存在的有效配对样本。"""
    dataset_root = dataset_root.resolve()
    image_root = resolve_path(image_dir, dataset_root)
    alpha_root = resolve_path(alpha_dir, dataset_root)
    image_index = index_by_relative_stem(iter_files(image_root, IMAGE_EXTENSIONS), image_root)
    alpha_index = index_by_relative_stem(iter_files(alpha_root, ALPHA_EXTENSIONS), alpha_root)
    common_keys = sorted(set(image_index) & set(alpha_index), key=natural_key)

    samples = []
    for key in common_keys:
        image_rel = image_index[key].relative_to(dataset_root).as_posix()
        alpha_rel = alpha_index[key].relative_to(dataset_root).as_posix()
        samples.append(PMSample(sample_id_from_relpath(image_rel), image_rel, alpha_rel))
    return samples


def load_split_entries(split_path: Path) -> List[PMSample]:
    """读取已有 split，供 append 模式判断哪些样本已经存在。"""
    samples = []
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t, ]+", line)
        if len(parts) < 2:
            raise ValueError(f"Expected image and alpha paths in split line: {raw_line}")
        image_rel, alpha_rel = parts[0], parts[1]
        samples.append(PMSample(sample_id_from_relpath(image_rel), image_rel, alpha_rel))
    return samples


def write_split(samples: Sequence[PMSample], split_path: Path) -> None:
    """把样本列表写成统一 split 文本，供后续模型评估复用。"""
    split_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PM alpha matting split",
        "# Format: image_rel_path<TAB>alpha_rel_path",
    ]
    lines.extend(f"{sample.image_rel}\t{sample.alpha_rel}" for sample in samples)
    split_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
