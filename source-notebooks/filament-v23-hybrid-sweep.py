from __future__ import annotations

"""Leakage-safe, spine-aware instance segmentation for MAGFiLO.

The detector is trained on consensus instances built across annotators.  A
crop refiner predicts both the filament mask and its annotated spine.  Every
validation prediction is produced by a fold that never saw the physical
observation, and scoring follows the organizer's August 2026 PQ notebook.
"""

import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


KAGGLE_DATA_AVAILABLE = Path("/kaggle/input").exists()
RUN_MODE = "proposal-v23-hybrid-sweep"
if KAGGLE_DATA_AVAILABLE:
    wheel = next(Path("/kaggle/input").glob("**/ultralytics-8.4.103-py3-none-any.whl"))
    dependency_dir = wheel.parent
    if importlib.util.find_spec("ultralytics") is None:
        wheels = [
            str(path)
            for pattern in (
                "ultralytics-*.whl",
                "ultralytics_thop-*.whl",
                "nvidia_ml_py-*.whl",
                "pycocotools-*.whl",
            )
            for path in sorted(dependency_dir.glob(pattern))
        ]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels]
        )
    torch_home = Path("/kaggle/working/torch_cache")
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resnet_checkpoint = checkpoint_dir / "resnet18-f37072fd.pth"
    if not resnet_checkpoint.exists():
        shutil.copy2(dependency_dir / "resnet18-f37072fd.pth", resnet_checkpoint)
    yolo_checkpoint = Path("/kaggle/working/yolo11s-seg.pt")
    if not yolo_checkpoint.exists():
        shutil.copy2(dependency_dir / "yolo11s-seg.pt", yolo_checkpoint)
    os.environ["TORCH_HOME"] = str(torch_home)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class CFG:
    seed: int = 20260814
    n_folds: int = 5
    debug: bool = False

    detector_model: str = "/kaggle/working/yolo11s-seg.pt"
    detector_size: int = 1280
    # Measured on Kaggle 2xT4: 30 epochs took about 48 minutes per fold.
    # Twenty epochs preserve the full five-fold ensemble while leaving enough
    # time for OOF/test inference inside a seven-hour session budget.
    detector_epochs: int = 20
    detector_batch: int = 4
    detector_patience: int = 7
    detector_conf: float = 0.18
    detector_fallback_conf: float = 0.05
    detector_iou: float = 0.60
    seed_nms_iou: float = 0.60
    detector_mask_ratio: int = 2

    consensus_match_iou: float = 0.15
    consensus_min_support: float = 0.50
    consensus_mask_threshold: float = 0.50

    refiner_crop: int = 384
    refiner_context: float = 1.8
    refiner_min_crop: int = 128
    refiner_epochs: int = 6
    # Per-GPU batch. The effective batch is doubled on Kaggle 2xT4.
    refiner_batch: int = 16
    refiner_lr: float = 3e-4
    refiner_threshold: float = 0.58
    spine_width: int = 5
    use_tta: bool = True

    minimum_area: int = 64
    pq_match_iou: float = 0.50
    ensemble_iou: float = 0.50
    minimum_fold_support: int = 2
    singleton_score: float = 0.92
    num_workers: int = 4
    runtime_target_hours: float = 7.0
    require_dual_gpu: bool = True


@dataclass(frozen=True)
class FusionPolicy:
    name: str
    match_iou: float
    minimum_support: int
    mask_mode: str
    minimum_area: int = 96
    minimum_clean_fraction: float = 0.65


RECALIBRATION_POLICIES = (
    FusionPolicy("s3_medoid", match_iou=0.20, minimum_support=3, mask_mode="medoid"),
    FusionPolicy("s3_majority", match_iou=0.20, minimum_support=3, mask_mode="majority"),
    FusionPolicy("s4_medoid", match_iou=0.20, minimum_support=4, mask_mode="medoid"),
)


@dataclass(frozen=True)
class AuditSetting:
    name: str
    detector_conf: float = 0.20
    seed_nms_iou: float = 0.60
    refiner_threshold: float = 0.65
    minimum_area: int = 96
    morphology: str = "none"


@dataclass(frozen=True)
class PretrainStrategy:
    name: str
    encoder_lr: float
    decoder_lr: float
    freeze_encoder_epochs: int = 0


PRETRAIN_STRATEGIES = (
    PretrainStrategy("full_low_lr", encoder_lr=1e-4, decoder_lr=1e-4),
    PretrainStrategy(
        "freeze_then_unfreeze", encoder_lr=5e-5, decoder_lr=3e-4,
        freeze_encoder_epochs=2,
    ),
    PretrainStrategy("discriminative_lr", encoder_lr=3e-5, decoder_lr=3e-4),
)


@dataclass(frozen=True)
class ProposalV2Setting:
    quality_threshold: float
    mask_threshold: float

    @property
    def name(self) -> str:
        return f"q{self.quality_threshold:.2f}_m{self.mask_threshold:.2f}"


@dataclass(frozen=True)
class ProposalV3Setting:
    existence_threshold: float
    score_threshold: float
    mask_threshold: float

    @property
    def name(self) -> str:
        return (
            f"e{self.existence_threshold:.2f}_q{self.score_threshold:.2f}_"
            f"m{self.mask_threshold:.2f}"
        )


def audit_settings() -> tuple[AuditSetting, ...]:
    """Small, targeted OOF sweep; each inference probability is computed once."""
    settings = [AuditSetting("baseline")]
    settings.extend(
        AuditSetting(f"conf_{value:.2f}", detector_conf=value)
        for value in (0.10, 0.15, 0.25, 0.30, 0.35)
    )
    settings.extend(
        AuditSetting(f"seed_nms_{value:.2f}", seed_nms_iou=value)
        for value in (0.45, 0.52, 0.68, 0.75)
    )
    settings.extend(
        AuditSetting(f"refiner_{value:.2f}", refiner_threshold=value)
        for value in (0.50, 0.55, 0.60, 0.70, 0.75)
    )
    settings.extend(
        AuditSetting(f"area_{value}", minimum_area=value)
        for value in (32, 64, 128, 192, 256)
    )
    settings.extend(
        AuditSetting(f"morph_{mode}", morphology=mode)
        for mode in ("open3", "close3", "close5", "open3_close3")
    )
    settings.extend(
        (
            AuditSetting(
                "balanced_recall", detector_conf=0.15, seed_nms_iou=0.68,
                refiner_threshold=0.60, minimum_area=64,
            ),
            AuditSetting(
                "balanced_precision", detector_conf=0.25, seed_nms_iou=0.52,
                refiner_threshold=0.65, minimum_area=128,
            ),
            AuditSetting(
                "thin_recall", detector_conf=0.20, seed_nms_iou=0.68,
                refiner_threshold=0.60, minimum_area=32, morphology="close3",
            ),
        )
    )
    return tuple(settings)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_paths() -> dict[str, Path]:
    annotations = sorted(
        Path("/kaggle/input").glob("**/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    )
    if len(annotations) != 1:
        raise RuntimeError(f"Expected exactly one competition annotation file, got {annotations}")
    annotation_path = annotations[0]
    root = annotation_path.parent.parent
    train = root / "train" / "train_images"
    test = root / "test" / "test_images"
    if not train.is_dir() or not test.is_dir():
        raise FileNotFoundError(f"Invalid competition data tree under {root}")
    return {"root": root, "json": annotation_path, "train": train, "test": test}


def build_indices(coco: Mapping[str, Any]) -> dict[str, Any]:
    images_by_id = {str(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[str, list[Mapping[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    image_ids_by_file: dict[str, list[str]] = {}
    for image_id, image in images_by_id.items():
        image_ids_by_file.setdefault(str(image["file_name"]), []).append(image_id)
    for image_ids in image_ids_by_file.values():
        image_ids.sort()
    return {
        "images_by_id": images_by_id,
        "annotations_by_image": annotations_by_image,
        "image_ids_by_file": image_ids_by_file,
    }


def annotation_to_rle(annotation: Mapping[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation.get("segmentation") or []
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        return mask_utils.frPyObjects(rle, height, width) if isinstance(rle.get("counts"), list) else rle
    if not segmentation:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), np.uint8)))
    return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))


def annotation_to_mask(annotation: Mapping[str, Any], height: int, width: int) -> np.ndarray:
    return mask_utils.decode(annotation_to_rle(annotation, height, width)).astype(bool)


def visualize_annotation_versions(
    image_dir: str | Path,
    indices: Mapping[str, Any],
    filenames: str | Path | Sequence[str | Path] | None = None,
    *,
    count: int = 3,
    seed: int = 20260814,
    only_multi_annotated: bool = True,
    mask_alpha: float = 0.32,
    show_spines: bool = True,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list[Any]:
    """Display every annotation version available for selected train images.

    One figure is created per physical image and one panel per COCO image ID.
    Consequently, repeated labels are never silently merged or omitted. When
    ``filenames`` is omitted, the selection is deterministic and, by default,
    restricted to images with multiple annotation versions.

    Args:
        image_dir: Directory containing the physical training JPEG files.
        indices: Result returned by :func:`build_indices`.
        filenames: One filename, or a sequence of filenames, to display.
        count: Number of files to sample when ``filenames`` is omitted.
        seed: Seed used for deterministic sampling.
        only_multi_annotated: Sample only files with at least two versions.
        mask_alpha: Opacity of the filled instance masks, between 0 and 1.
        show_spines: Draw the annotated spine for every instance when present.
        save_dir: Optional directory in which each figure is saved as PNG.
        show: Call ``matplotlib.pyplot.show`` for notebook display.

    Returns:
        The list of Matplotlib figures, in the requested/sample order.
    """
    import matplotlib.pyplot as plt

    if count < 1:
        raise ValueError("count must be at least 1")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be between 0 and 1")

    versions_by_file = indices["image_ids_by_file"]
    if filenames is None:
        candidates = sorted(
            filename
            for filename, image_ids in versions_by_file.items()
            if not only_multi_annotated or len(image_ids) > 1
        )
        if not candidates:
            qualifier = " with multiple annotation versions" if only_multi_annotated else ""
            raise ValueError(f"No training images found{qualifier}")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
        selected = [str(filename) for filename in chosen]
    else:
        requested = [filenames] if isinstance(filenames, (str, Path)) else list(filenames)
        if not requested:
            raise ValueError("filenames must not be empty")
        selected = [Path(filename).name for filename in requested]

    unknown = [filename for filename in selected if filename not in versions_by_file]
    if unknown:
        preview = ", ".join(unknown[:5])
        suffix = " ..." if len(unknown) > 5 else ""
        raise KeyError(f"Unknown training filename(s): {preview}{suffix}")

    source_dir = Path(image_dir)
    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Any] = []
    for filename in selected:
        image_path = source_dir / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read training image: {image_path}")

        version_ids = list(versions_by_file[filename])
        figure, axes = plt.subplots(
            1,
            len(version_ids),
            figsize=(5.2 * len(version_ids), 5.5),
            squeeze=False,
            constrained_layout=True,
        )
        for version_index, (axis, image_id) in enumerate(zip(axes[0], version_ids), start=1):
            info = indices["images_by_id"][image_id]
            height, width = int(info["height"]), int(info["width"])
            annotations = [
                annotation
                for annotation in indices["annotations_by_image"].get(image_id, [])
                if not annotation.get("iscrowd", 0)
            ]
            palette = plt.get_cmap("turbo", max(1, len(annotations)))

            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            overlay = np.zeros((height, width, 4), dtype=np.uint8)
            for annotation_index, annotation in enumerate(annotations):
                mask = annotation_to_mask(annotation, height, width)
                color = palette(annotation_index)
                overlay[mask, :3] = np.round(np.asarray(color[:3]) * 255).astype(np.uint8)
                overlay[mask, 3] = round(mask_alpha * 255)
                if mask.any():
                    axis.contour(
                        mask.astype(np.uint8),
                        levels=[0.5],
                        colors=[color],
                        linewidths=0.8,
                        zorder=2,
                    )

                spine = np.asarray(annotation.get("spine") or [], dtype=float)
                if show_spines and spine.size >= 4 and spine.size % 2 == 0:
                    points = spine.reshape(-1, 2)
                    axis.plot(
                        points[:, 0], points[:, 1], color="black", linewidth=2.4, zorder=3
                    )
                    axis.plot(
                        points[:, 0], points[:, 1], color=color, linewidth=1.25, zorder=4
                    )
            axis.imshow(overlay, zorder=1)

            record_code = image_id.split("-", 1)[0]
            axis.set_title(
                f"Version {version_index}/{len(version_ids)} · {record_code}\n"
                f"{len(annotations)} instance(s)",
                fontsize=10,
            )
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
            axis.axis("off")

        date_captured = indices["images_by_id"][version_ids[0]].get("date_captured", "")
        figure.suptitle(
            f"{filename} · {date_captured} · {len(version_ids)} annotation version(s)\n"
            "colored area = mask · colored centerline = spine",
            fontsize=12,
        )
        if output_dir is not None:
            figure.savefig(
                output_dir / f"{Path(filename).stem}__all_annotation_versions.png",
                dpi=160,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        figures.append(figure)

    return figures


def semantic_rle(image_id: str, indices: Mapping[str, Any]) -> dict[str, Any]:
    info = indices["images_by_id"][image_id]
    rles = [
        annotation_to_rle(annotation, int(info["height"]), int(info["width"]))
        for annotation in indices["annotations_by_image"].get(image_id, [])
        if not annotation.get("iscrowd", 0)
    ]
    if rles:
        return mask_utils.merge(rles)
    return mask_utils.encode(
        np.asfortranarray(np.zeros((int(info["height"]), int(info["width"])), np.uint8))
    )


def choose_medoid_records(indices: Mapping[str, Any]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for filename, image_ids in indices["image_ids_by_file"].items():
        if len(image_ids) == 1:
            selected[filename] = image_ids[0]
            continue
        rles = [semantic_rle(image_id, indices) for image_id in image_ids]
        pairwise = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=float)
        agreement = (pairwise.sum(axis=1) - 1.0) / (len(rles) - 1)
        selected[filename] = image_ids[int(np.argmax(agreement))]
    return selected


def match_annotations(
    anchors: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    info: Mapping[str, Any],
    minimum_iou: float,
) -> tuple[list[tuple[int, int]], list[int]]:
    if not anchors or not candidates:
        return [], list(range(len(candidates)))
    height, width = int(info["height"]), int(info["width"])
    anchor_rles = [annotation_to_rle(annotation, height, width) for annotation in anchors]
    candidate_rles = [annotation_to_rle(annotation, height, width) for annotation in candidates]
    overlaps = np.asarray(mask_utils.iou(anchor_rles, candidate_rles, [0] * len(candidate_rles)))
    anchor_indices, candidate_indices = linear_sum_assignment(-overlaps)
    matches = [
        (int(anchor_index), int(candidate_index))
        for anchor_index, candidate_index in zip(anchor_indices, candidate_indices)
        if overlaps[anchor_index, candidate_index] >= minimum_iou
    ]
    matched_candidates = {candidate_index for _, candidate_index in matches}
    return matches, [index for index in range(len(candidates)) if index not in matched_candidates]


def build_consensus_catalog(
    indices: Mapping[str, Any], medoids: Mapping[str, str], cfg: CFG
) -> dict[str, list[dict[str, Any]]]:
    """Cluster instance annotations while retaining annotator uncertainty."""
    catalog: dict[str, list[dict[str, Any]]] = {}
    for filename, image_ids in sorted(indices["image_ids_by_file"].items()):
        medoid_id = medoids[filename]
        info = indices["images_by_id"][medoid_id]
        medoid_annotations = [
            annotation
            for annotation in indices["annotations_by_image"].get(medoid_id, [])
            if not annotation.get("iscrowd", 0)
        ]
        clusters = [
            {"members": [(medoid_id, annotation)], "representative": annotation}
            for annotation in medoid_annotations
        ]
        for image_id in image_ids:
            if image_id == medoid_id:
                continue
            candidates = [
                annotation
                for annotation in indices["annotations_by_image"].get(image_id, [])
                if not annotation.get("iscrowd", 0)
            ]
            representatives = [cluster["representative"] for cluster in clusters]
            matches, unmatched = match_annotations(
                representatives, candidates, info, cfg.consensus_match_iou
            )
            for cluster_index, candidate_index in matches:
                clusters[cluster_index]["members"].append((image_id, candidates[candidate_index]))
            for candidate_index in unmatched:
                clusters.append(
                    {
                        "members": [(image_id, candidates[candidate_index])],
                        "representative": candidates[candidate_index],
                    }
                )
        minimum_support = max(1, math.ceil(len(image_ids) * cfg.consensus_min_support))
        retained = []
        for cluster in clusters:
            support = len({image_id for image_id, _ in cluster["members"]})
            contains_medoid = any(image_id == medoid_id for image_id, _ in cluster["members"])
            if support >= minimum_support or contains_medoid:
                retained.append(
                    {
                        "members": cluster["members"],
                        "support": support / len(image_ids),
                        "n_annotators": len(image_ids),
                    }
                )
        catalog[filename] = retained
    return catalog


def build_observations(indices: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for filename, image_ids in sorted(indices["image_ids_by_file"].items()):
        counts = [len(indices["annotations_by_image"].get(image_id, [])) for image_id in image_ids]
        rows.append(
            {
                "filename": filename,
                "site": Path(filename).stem[-2:],
                "year": Path(filename).stem[:4],
                "n_annotators": len(image_ids),
                "mean_instances": float(np.mean(counts)),
            }
        )
    return pd.DataFrame(rows)


def assign_folds(observations: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    frame = observations.copy().sort_values("filename").reset_index(drop=True)
    count_bin = pd.qcut(
        frame["mean_instances"].rank(method="first"),
        q=min(5, len(frame)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    frame["stratum"] = frame["site"] + "_q" + count_bin.astype(str)
    frame["fold"] = -1
    fold_sizes = np.zeros(cfg.n_folds, dtype=int)
    rng = np.random.default_rng(cfg.seed)
    groups = sorted(frame.groupby("stratum").groups.items(), key=lambda item: -len(item[1]))
    for _, row_indices in groups:
        row_indices = np.asarray(list(row_indices), dtype=int)
        rng.shuffle(row_indices)
        for row_index in row_indices:
            choices = np.flatnonzero(fold_sizes == fold_sizes.min())
            fold = int(rng.choice(choices))
            frame.loc[row_index, "fold"] = fold
            fold_sizes[fold] += 1
    assert frame["filename"].is_unique
    assert (frame["fold"] >= 0).all()
    return frame


def decode_consensus(
    cluster: Mapping[str, Any], indices: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    spines = []
    for image_id, annotation in cluster["members"]:
        info = indices["images_by_id"][image_id]
        height, width = int(info["height"]), int(info["width"])
        masks.append(annotation_to_mask(annotation, height, width).astype(np.float32))
        spine = np.zeros((height, width), np.uint8)
        points = np.asarray(annotation.get("spine") or [], dtype=np.float32).reshape(-1, 2)
        if len(points) >= 2:
            cv2.polylines(
                spine,
                [np.round(points).astype(np.int32)],
                False,
                1,
                thickness=CFG.spine_width,
                lineType=cv2.LINE_AA,
            )
        spines.append(spine.astype(np.float32))
    return np.mean(masks, axis=0), np.mean(spines, axis=0)


def mask_to_polygon(mask: np.ndarray) -> str | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    height, width = mask.shape
    contour = contour.astype(np.float32)
    contour[:, 0] = np.clip(contour[:, 0] / width, 0, 1)
    contour[:, 1] = np.clip(contour[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in contour.ravel())


def write_detector_split(
    output: Path,
    filenames: Sequence[str],
    image_dir: Path,
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    indices: Mapping[str, Any],
    cfg: CFG,
) -> None:
    images_dir, labels_dir = output / "images", output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        destination = images_dir / filename
        if not destination.exists():
            os.symlink(image_dir / filename, destination)
        labels = []
        for cluster in catalog[filename]:
            mask_probability, _ = decode_consensus(cluster, indices)
            hard_mask = mask_probability >= cfg.consensus_mask_threshold
            polygon = mask_to_polygon(hard_mask)
            if polygon is not None:
                labels.append(polygon)
        (labels_dir / f"{Path(filename).stem}.txt").write_text("\n".join(labels), encoding="utf-8")


def prepare_detector_fold(
    fold: int,
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> Path:
    fold_dir = work_dir / "detector_data" / f"fold_{fold}"
    yaml_path = fold_dir / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    train_files = manifest.loc[manifest["fold"] != fold, "filename"].tolist()
    val_files = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
    if cfg.debug:
        train_files, val_files = train_files[:32], val_files[:12]
    write_detector_split(fold_dir / "train", train_files, paths["train"], catalog, indices, cfg)
    write_detector_split(fold_dir / "val", val_files, paths["train"], catalog, indices, cfg)
    yaml_path.write_text(
        f"path: {fold_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    return yaml_path


def train_detector(fold: int, yaml_path: Path, cfg: CFG, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model_dir = work_dir / "models" / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "detector.pt"
    if checkpoint.exists():
        return checkpoint
    model = YOLO(cfg.detector_model)
    gpu_count = torch.cuda.device_count()
    device: int | str = ",".join(map(str, range(gpu_count))) if gpu_count > 1 else (0 if gpu_count else "cpu")
    print(
        {
            "stage": "detector",
            "fold": fold,
            "cuda_devices": list(range(gpu_count)),
            "ultralytics_device": device,
            "epochs": 1 if cfg.debug else cfg.detector_epochs,
        }
    )
    model.train(
        data=str(yaml_path),
        epochs=1 if cfg.debug else cfg.detector_epochs,
        imgsz=640 if cfg.debug else cfg.detector_size,
        batch=min(cfg.detector_batch, 4) if cfg.debug else cfg.detector_batch,
        device=device,
        patience=cfg.detector_patience,
        seed=cfg.seed + fold,
        single_cls=True,
        mask_ratio=cfg.detector_mask_ratio,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        degrees=15,
        fliplr=0.5,
        flipud=0.5,
        scale=0.30,
        translate=0.08,
        mosaic=0.25,
        close_mosaic=5,
        project=str(work_dir / "detector_runs"),
        name=f"fold_{fold}",
        exist_ok=True,
        verbose=True,
    )
    shutil.copy2(str(model.trainer.best), checkpoint)
    detector_history = Path(model.trainer.save_dir) / "results.csv"
    if detector_history.is_file():
        shutil.copy2(detector_history, model_dir / "detector_history.csv")
    return checkpoint


def square_bounds(mask: np.ndarray, context: float, minimum: int) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    center_x, center_y = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(np.ptp(xs) + 1, np.ptp(ys) + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return x0, y0, x1, y1


def jitter_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seed = mask.astype(np.uint8)
    radius = int(rng.integers(-7, 8))
    if radius:
        kernel_size = 2 * abs(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel)
    shift_x, shift_y = int(rng.integers(-12, 13)), int(rng.integers(-12, 13))
    seed = cv2.warpAffine(
        seed,
        np.float32([[1, 0, shift_x], [0, 1, shift_y]]),
        seed.shape[::-1],
        flags=cv2.INTER_NEAREST,
    )
    return seed.astype(bool) if seed.any() else mask.astype(bool)


class ConsensusRefinerDataset(Dataset):
    def __init__(
        self,
        filenames: Sequence[str],
        image_dir: Path,
        catalog: Mapping[str, Sequence[Mapping[str, Any]]],
        indices: Mapping[str, Any],
        cfg: CFG,
        augment: bool,
    ) -> None:
        self.image_dir = image_dir
        self.catalog = catalog
        self.indices = indices
        self.cfg = cfg
        self.augment = augment
        self.samples = [(filename, cluster) for filename in filenames for cluster in catalog[filename]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filename, cluster = self.samples[index]
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        mask_probability, spine_probability = decode_consensus(cluster, self.indices)
        hard_mask = mask_probability >= self.cfg.consensus_mask_threshold
        rng = np.random.default_rng(self.cfg.seed + index + random.randint(0, 1_000_000))
        seed = jitter_mask(hard_mask, rng) if self.augment else hard_mask
        x0, y0, x1, y1 = square_bounds(seed, self.cfg.refiner_context, self.cfg.refiner_min_crop)
        size = (self.cfg.refiner_crop, self.cfg.refiner_crop)
        image_crop = cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
        seed_crop = cv2.resize(seed[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        mask_crop = cv2.resize(mask_probability[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
        spine_crop = cv2.resize(spine_probability[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
        if self.augment:
            k = random.randrange(4)
            features = np.rot90(features, k, axes=(1, 2)).copy()
            mask_crop = np.rot90(mask_crop, k).copy()
            spine_crop = np.rot90(spine_crop, k).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                mask_crop = mask_crop[:, ::-1].copy()
                spine_crop = spine_crop[:, ::-1].copy()
            features[:2] = np.clip(features[:2] * random.uniform(0.85, 1.15), 0, 1)
        return (
            torch.from_numpy(features),
            torch.from_numpy(mask_crop[None].astype(np.float32)),
            torch.from_numpy(spine_crop[None].astype(np.float32)),
            torch.tensor(float(cluster["support"]), dtype=torch.float32),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class SpineAwareRefiner(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.mask_head = nn.Conv2d(16, 1, 1)
        self.spine_head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor)
        x1 = self.layer1(self.pool(x0))
        x2, x3 = self.layer2(x1), None
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        features = self.up0(x)
        return self.mask_head(features), self.spine_head(features)


class LegacyConvBlock(nn.Sequential):
    """Exact decoder block used by the public new-metric baseline."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class LegacySeedRefiner(nn.Module):
    """Inference-compatible architecture for the saved baseline checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), LegacyConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), LegacyConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), LegacyConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), LegacyConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        return self.head(self.up0(x))


class LegacyRefinerDataset(Dataset):
    """Exact supervised crop task used by the 0.36 public baseline."""

    def __init__(
        self,
        filenames: Sequence[str],
        image_dir: Path,
        selected_image_ids: Mapping[str, str],
        indices: Mapping[str, Any],
        cfg: CFG,
        augment: bool,
    ) -> None:
        self.image_dir = image_dir
        self.indices = indices
        self.cfg = cfg
        self.augment = augment
        self.samples: list[tuple[str, str, Mapping[str, Any]]] = []
        for filename in filenames:
            image_id = selected_image_ids[filename]
            for annotation in indices["annotations_by_image"].get(image_id, []):
                if not annotation.get("iscrowd", 0):
                    self.samples.append((filename, image_id, annotation))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, image_id, annotation = self.samples[index]
        info = self.indices["images_by_id"][image_id]
        image = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        target = annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
        rng = np.random.default_rng(self.cfg.seed + index + random.randint(0, 1_000_000))
        seed = target.astype(np.uint8)
        if self.augment:
            radius = int(rng.integers(-5, 6))
            if radius:
                width = 2 * abs(radius) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
                seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel)
            shift_x, shift_y = int(rng.integers(-8, 9)), int(rng.integers(-8, 9))
            seed = cv2.warpAffine(
                seed,
                np.float32([[1, 0, shift_x], [0, 1, shift_y]]),
                seed.shape[::-1],
                flags=cv2.INTER_NEAREST,
            )
            if not seed.any():
                seed = target.astype(np.uint8)
        seed_bool = seed.astype(bool)
        x0, y0, x1, y1 = square_bounds(
            seed_bool, self.cfg.refiner_context, self.cfg.refiner_min_crop
        )
        size = (self.cfg.refiner_crop,) * 2
        image_crop = cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
        target_crop = cv2.resize(
            target[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
        if self.augment:
            rotation = random.randrange(4)
            features = np.rot90(features, rotation, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, rotation).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target_crop = target_crop[:, ::-1].copy()
            features[:2] = np.clip(features[:2] * random.uniform(0.9, 1.1), 0, 1)
        return (
            torch.from_numpy(features),
            torch.from_numpy(target_crop[None].astype(np.float32)),
        )


def discover_domain_backbone() -> Path | None:
    candidates = sorted(
        Path("/kaggle/input").glob("**/resnet18_gong_simsiam.pt")
    ) if Path("/kaggle/input").is_dir() else []
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple GONG pretrained backbones found: {candidates}")
    return candidates[0] if candidates else None


def load_domain_backbone(model: LegacySeedRefiner, checkpoint: Path) -> dict[str, Any]:
    source = torch.load(checkpoint, map_location="cpu")
    if not isinstance(source, Mapping):
        raise TypeError(f"Expected a state dict in {checkpoint}")
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for source_key, value in source.items():
        if source_key == "conv1.weight":
            target_key = "stem.0.weight"
        elif source_key.startswith("bn1."):
            target_key = "stem.1." + source_key.removeprefix("bn1.")
        elif source_key.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
            target_key = source_key
        else:
            continue
        if target_key not in target:
            raise KeyError(f"Unknown mapped encoder key: {target_key}")
        if target[target_key].shape != value.shape:
            raise ValueError(
                f"Backbone shape mismatch for {source_key}: {value.shape} != {target[target_key].shape}"
            )
        mapped[target_key] = value
    expected = {
        key
        for key in target
        if key.startswith(("stem.0.", "stem.1.", "layer1.", "layer2.", "layer3.", "layer4."))
    }
    missing = sorted(expected - set(mapped))
    if missing:
        raise KeyError(f"Missing {len(missing)} encoder tensors, first={missing[:5]}")
    target.update(mapped)
    model.load_state_dict(target, strict=True)
    return {
        "checkpoint": str(checkpoint),
        "loaded_tensors": len(mapped),
        "ignored_source_tensors": len(source) - len(mapped),
    }


def validate_dual_gpu_refiner() -> None:
    """Fail fast unless a refiner forward/backward really reaches both GPUs."""
    if torch.cuda.device_count() < 2:
        return
    device = torch.device("cuda:0")
    base_model = SpineAwareRefiner(pretrained=False).to(device).train()
    devices_seen: set[int] = set()

    def record_device(_module: nn.Module, _inputs: Any, outputs: Any) -> None:
        index = outputs[0].device.index
        if index is not None:
            devices_seen.add(index)

    hook = base_model.register_forward_hook(record_device)
    parallel_model = nn.DataParallel(base_model, device_ids=[0, 1])
    dummy = torch.zeros((4, 3, 64, 64), device=device)
    with torch.amp.autocast("cuda"):
        mask_logits, spine_logits = parallel_model(dummy)
    loss = mask_logits.mean() + spine_logits.mean()
    loss.backward()
    torch.cuda.synchronize()
    hook.remove()
    if devices_seen != {0, 1}:
        raise RuntimeError(f"Dual-GPU refiner preflight failed: devices_seen={sorted(devices_seen)}")
    print({"stage": "dual_gpu_preflight", "cuda_devices_seen": sorted(devices_seen)})
    del dummy, parallel_model, base_model, mask_logits, spine_logits, loss
    gc.collect()
    torch.cuda.empty_cache()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def spine_coverage_loss(mask_logits: torch.Tensor, spine_targets: torch.Tensor) -> torch.Tensor:
    probabilities = mask_logits.sigmoid()
    covered = (probabilities * spine_targets).sum(dim=(1, 2, 3))
    target = spine_targets.sum(dim=(1, 2, 3))
    return (1 - (covered + 1) / (target + 1)).mean()


def train_refiner(
    fold: int,
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> Path:
    model_dir = work_dir / "models" / f"fold_{fold}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "spine_refiner.pt"
    if checkpoint.exists():
        return checkpoint
    train_files = manifest.loc[manifest["fold"] != fold, "filename"].tolist()
    val_files = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
    if cfg.debug:
        train_files, val_files = train_files[:32], val_files[:12]
    train_dataset = ConsensusRefinerDataset(train_files, paths["train"], catalog, indices, cfg, True)
    val_dataset = ConsensusRefinerDataset(val_files, paths["train"], catalog, indices, cfg, False)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_batch = (
        min(cfg.refiner_batch, 4)
        if cfg.debug
        else cfg.refiner_batch * max(gpu_count, 1)
    )
    loader_options = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg.refiner_batch, 8) if cfg.debug else effective_batch * 2,
        shuffle=False,
        **loader_options,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpineAwareRefiner(pretrained=True).to(device)
    training_model: nn.Module = model
    if gpu_count > 1:
        training_model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    print(
        {
            "stage": "refiner",
            "fold": fold,
            "cuda_devices": list(range(gpu_count)),
            "parallel": "DataParallel" if gpu_count > 1 else "single-device",
            "batch_per_gpu": cfg.refiner_batch,
            "effective_batch": effective_batch,
            "epochs": 1 if cfg.debug else cfg.refiner_epochs,
        }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.refiner_lr, weight_decay=1e-4)
    epochs = 1 if cfg.debug else cfg.refiner_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_score = -1.0
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        training_model.train()
        running_loss = 0.0
        for features, mask_targets, spine_targets, support in train_loader:
            features = features.to(device, non_blocking=True)
            mask_targets = mask_targets.to(device, non_blocking=True)
            spine_targets = spine_targets.to(device, non_blocking=True)
            support = support.to(device, non_blocking=True).view(-1, 1, 1, 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                mask_logits, spine_logits = training_model(features)
                mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_targets, reduction="none")
                mask_bce = (mask_bce * (0.5 + 0.5 * support)).mean()
                loss = (
                    0.30 * mask_bce
                    + 0.35 * soft_dice_loss(mask_logits, mask_targets)
                    + 0.12 * F.binary_cross_entropy_with_logits(spine_logits, spine_targets)
                    + 0.13 * soft_dice_loss(spine_logits, spine_targets)
                    + 0.10 * spine_coverage_loss(mask_logits, spine_targets)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
        training_model.eval()
        dice_total = spine_total = sample_count = 0
        with torch.no_grad():
            for features, mask_targets, spine_targets, _ in val_loader:
                features = features.to(device)
                mask_targets = mask_targets.to(device) >= 0.5
                spine_targets = spine_targets.to(device) >= 0.5
                mask_logits, spine_logits = training_model(features)
                mask_predictions = mask_logits.sigmoid() >= cfg.refiner_threshold
                spine_predictions = spine_logits.sigmoid() >= 0.5
                for predictions, targets, accumulator in (
                    (mask_predictions, mask_targets, "mask"),
                    (spine_predictions, spine_targets, "spine"),
                ):
                    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
                    denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
                    values = ((2 * intersection + 1) / (denominator + 1)).sum().item()
                    if accumulator == "mask":
                        dice_total += values
                    else:
                        spine_total += values
                sample_count += len(features)
        mask_dice = dice_total / max(sample_count, 1)
        spine_dice = spine_total / max(sample_count, 1)
        score = 0.8 * mask_dice + 0.2 * spine_dice
        epoch_seconds = time.monotonic() - epoch_started
        print(
            f"fold={fold} epoch={epoch + 1} loss={running_loss / max(len(train_loader), 1):.5f} "
            f"mask_dice={mask_dice:.5f} spine_dice={spine_dice:.5f} "
            f"seconds={epoch_seconds:.1f}"
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    return checkpoint


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def resolve_overlaps(
    masks: Sequence[np.ndarray], scores: Sequence[float], minimum_area: int
) -> tuple[list[np.ndarray], list[float]]:
    if not masks:
        return [], []
    assigned = np.zeros_like(masks[0], dtype=bool)
    output_masks, output_scores = [], []
    for index in np.argsort(scores)[::-1]:
        clean = masks[int(index)].astype(bool) & ~assigned
        if clean.sum() >= minimum_area:
            output_masks.append(clean)
            output_scores.append(float(scores[int(index)]))
            assigned |= clean
    return output_masks, output_scores


def remove_duplicate_seeds(
    masks: Sequence[np.ndarray], scores: Sequence[float], threshold: float = 0.60
) -> tuple[list[np.ndarray], list[float]]:
    kept_masks, kept_scores = [], []
    for index in np.argsort(scores)[::-1]:
        mask = masks[int(index)]
        if all(mask_iou(mask, kept) < threshold for kept in kept_masks):
            kept_masks.append(mask)
            kept_scores.append(float(scores[int(index)]))
    return kept_masks, kept_scores


@torch.no_grad()
def refine_seed(
    image: np.ndarray,
    seed_mask: np.ndarray,
    model: SpineAwareRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = square_bounds(seed_mask, cfg.refiner_context, cfg.refiner_min_crop)
    size = (cfg.refiner_crop, cfg.refiner_crop)
    image_crop = cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
    seed_crop = cv2.resize(seed_mask[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
    features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(features[None]).to(device)
    transforms = [lambda value: value]
    inverses = [lambda value: value]
    if cfg.use_tta:
        transforms += [lambda value: value.flip(-1), lambda value: value.flip(-2)]
        inverses += [lambda value: value.flip(-1), lambda value: value.flip(-2)]
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        mask_logits, spine_logits = model(transformed)
    mask_probabilities, spine_probabilities = [], []
    for index, inverse in enumerate(inverses):
        mask_probabilities.append(
            inverse(mask_logits[index : index + 1].sigmoid())[0, 0].float().cpu().numpy()
        )
        spine_probabilities.append(
            inverse(spine_logits[index : index + 1].sigmoid())[0, 0].float().cpu().numpy()
        )
    mask_probability = np.mean(mask_probabilities, axis=0)
    spine_probability = np.mean(spine_probabilities, axis=0)
    resized = cv2.resize(mask_probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    crop_mask = resized >= cfg.refiner_threshold
    full_mask = np.zeros_like(seed_mask, dtype=bool)
    full_mask[y0:y1, x0:x1] = crop_mask
    if not full_mask.any():
        full_mask = seed_mask.astype(bool)
    mask_confidence = float(resized[crop_mask].mean()) if crop_mask.any() else 0.0
    spine_confidence = float(np.mean(spine_probability[spine_probability >= 0.5])) if np.any(spine_probability >= 0.5) else 0.0
    return full_mask, 0.8 * mask_confidence + 0.2 * spine_confidence


def fold_device(fold: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{fold % torch.cuda.device_count()}")


def load_models(fold: int, cfg: CFG, work_dir: Path) -> tuple[Any, SpineAwareRefiner, torch.device]:
    from ultralytics import YOLO

    model_dir = work_dir / "models" / f"fold_{fold}"
    detector = YOLO(str(model_dir / "detector.pt"))
    device = fold_device(fold)
    refiner = SpineAwareRefiner(pretrained=False)
    refiner.load_state_dict(torch.load(model_dir / "spine_refiner.pt", map_location=device))
    return detector, refiner.to(device).eval(), device


def discover_parent_checkpoint_roots() -> dict[str, Path]:
    """Locate reusable checkpoints exposed as Kaggle kernel inputs."""
    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        return {}
    patterns = {
        "spine": "**/spine_consensus/models/fold_0/spine_refiner.pt",
        "baseline": "**/solar_filament_pq/models/fold_0/refiner.pt",
    }
    roots: dict[str, Path] = {}
    for family, pattern in patterns.items():
        candidates = sorted({path.parents[2] for path in input_root.glob(pattern)})
        valid = [
            root
            for root in candidates
            if all(
                (root / "models" / f"fold_{fold}" / "detector.pt").is_file()
                for fold in range(5)
            )
        ]
        if len(valid) > 1:
            raise RuntimeError(f"Multiple {family} checkpoint roots found: {valid}")
        if valid:
            roots[family] = valid[0]
    return roots


def legacy_inference_cfg() -> CFG:
    """Settings recorded in the successful public new-metric baseline log."""
    return CFG(
        seed=20260729,
        detector_size=1024,
        detector_epochs=30,
        detector_conf=0.20,
        detector_fallback_conf=0.05,
        detector_iou=0.60,
        seed_nms_iou=0.45,
        refiner_crop=256,
        refiner_context=1.8,
        refiner_min_crop=96,
        refiner_epochs=8,
        refiner_batch=16,
        refiner_threshold=0.65,
        minimum_area=96,
    )


def load_legacy_models(
    fold: int, cfg: CFG, work_dir: Path
) -> tuple[Any, LegacySeedRefiner, torch.device]:
    from ultralytics import YOLO

    model_dir = work_dir / "models" / f"fold_{fold}"
    detector = YOLO(str(model_dir / "detector.pt"))
    device = fold_device(fold)
    refiner = LegacySeedRefiner()
    refiner.load_state_dict(torch.load(model_dir / "refiner.pt", map_location=device))
    return detector, refiner.to(device).eval(), device


def component_count(mask: np.ndarray) -> int:
    return max(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1, 0)


def binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    return float((2 * intersection + 1e-7) / (first.sum() + second.sum() + 1e-7))


@torch.no_grad()
def refine_legacy_seed(
    image: np.ndarray,
    seed_mask: np.ndarray,
    model: LegacySeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[np.ndarray, float]:
    """Reproduce baseline refinement while batching its four TTA transforms."""
    x0, y0, x1, y1 = square_bounds(seed_mask, cfg.refiner_context, cfg.refiner_min_crop)
    size = (cfg.refiner_crop, cfg.refiner_crop)
    image_crop = cv2.resize(image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
    seed_crop = cv2.resize(
        seed_mask[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
    features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(features[None]).to(device)
    transforms = [
        lambda value: value,
        lambda value: value.flip(-1),
        lambda value: value.flip(-2),
        lambda value: value.flip((-2, -1)),
    ]
    inverses = transforms
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    logits = model(transformed)
    probabilities = [
        inverse(logits[index : index + 1].sigmoid())[0, 0].float().cpu().numpy()
        for index, inverse in enumerate(inverses)
    ]
    base_probability = probabilities[0]
    candidate_probability = np.mean(probabilities, axis=0)
    base_mask = base_probability >= cfg.refiner_threshold
    candidate_mask = candidate_probability >= cfg.refiner_threshold
    use_candidate = True
    if base_mask.any():
        area_ratio = candidate_mask.sum() / max(base_mask.sum(), 1)
        use_candidate = (
            0.60 <= area_ratio <= 1.70
            and binary_dice(base_mask, candidate_mask) >= 0.62
            and component_count(candidate_mask) <= component_count(base_mask) + 1
        )
    probability = candidate_probability if use_candidate else base_probability
    resized = cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    crop_mask = resized >= cfg.refiner_threshold
    full_mask = np.zeros_like(seed_mask, dtype=bool)
    full_mask[y0:y1, x0:x1] = crop_mask
    if not full_mask.any():
        full_mask = seed_mask.astype(bool)
    confidence = float(resized[crop_mask].mean()) if crop_mask.any() else 0.0
    return full_mask, confidence


@torch.no_grad()
def predict_legacy_instances(
    image_path: Path,
    detector: Any,
    refiner: LegacySeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[list[np.ndarray], list[float]]:
    # The OOF sweep showed that the detector already proposes nearly every
    # filament, while duplicate proposals dominate its false positives.  Keep
    # the high-recall proposal pass used by the audit and apply the validated
    # mask-level NMS before refinement.
    image, seeds, detector_scores = raw_legacy_candidates(
        image_path, detector, device, cfg
    )
    if not seeds:
        seed = classical_fallback_seed(image, cfg.minimum_area)
        mask, quality = refine_legacy_seed(image, seed, refiner, device, cfg)
        return [mask], [0.01 * quality]
    eligible = [
        index
        for index, score in enumerate(detector_scores)
        if score >= cfg.detector_conf
    ]
    if not eligible:
        eligible = list(range(len(seeds)))
    records = legacy_probability_records(
        image,
        [seeds[index] for index in eligible],
        [detector_scores[index] for index in eligible],
        refiner,
        device,
        cfg,
    )
    setting = AuditSetting(
        "production",
        detector_conf=0.0,
        seed_nms_iou=cfg.seed_nms_iou,
        refiner_threshold=cfg.refiner_threshold,
        minimum_area=cfg.minimum_area,
    )
    return predictions_for_setting(records, setting)


def classical_fallback_seed(image: np.ndarray, minimum_area: int) -> np.ndarray:
    """Image-only fallback used when the detector returns no instance."""
    blurred = cv2.GaussianBlur(image, (0, 0), 25)
    darkness = blurred.astype(np.float32) - image.astype(np.float32)
    disk = image > 12
    values = darkness[disk]
    if values.size == 0:
        disk = np.ones_like(image, dtype=bool)
        values = darkness.ravel()
    candidate = ((darkness >= float(np.quantile(values, 0.995))) & disk).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    valid = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area
    ]
    if valid:
        return labels == max(valid, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    fallback = np.zeros_like(image, dtype=np.uint8)
    center_y, center_x = np.unravel_index(int(np.argmax(darkness)), darkness.shape)
    cv2.circle(fallback, (int(center_x), int(center_y)), 6, 1, -1)
    return fallback.astype(bool)


@torch.no_grad()
def predict_instances(
    image_path: Path,
    detector: Any,
    refiner: SpineAwareRefiner,
    device: torch.device,
    cfg: CFG,
) -> tuple[list[np.ndarray], list[float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    result = detector.predict(
        str(image_path),
        imgsz=640 if cfg.debug else cfg.detector_size,
        conf=cfg.detector_conf,
        iou=cfg.detector_iou,
        agnostic_nms=True,
        retina_masks=True,
        device=device.index if device.type == "cuda" else "cpu",
        verbose=False,
    )[0]
    if result.masks is None:
        result = detector.predict(
            str(image_path),
            imgsz=640 if cfg.debug else cfg.detector_size,
            conf=cfg.detector_fallback_conf,
            iou=cfg.detector_iou,
            agnostic_nms=True,
            retina_masks=True,
            device=device.index if device.type == "cuda" else "cpu",
            verbose=False,
        )[0]
    if result.masks is None:
        seed = classical_fallback_seed(image, cfg.minimum_area)
        mask, quality = refine_seed(image, seed, refiner, device, cfg)
        return [mask], [0.01 * quality]
    seeds = [mask.cpu().numpy().astype(bool) for mask in result.masks.data]
    if seeds and seeds[0].shape != image.shape:
        seeds = [
            cv2.resize(seed.astype(np.uint8), image.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            for seed in seeds
        ]
    detector_scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    seeds, detector_scores = remove_duplicate_seeds(seeds, detector_scores)
    masks, scores = [], []
    for seed, detector_score in zip(seeds, detector_scores):
        if seed.sum() < cfg.minimum_area:
            continue
        mask, quality = refine_seed(image, seed, refiner, device, cfg)
        masks.append(mask)
        scores.append(float(detector_score * quality))
    return resolve_overlaps(masks, scores, cfg.minimum_area)


def pairwise_iou(gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray]) -> np.ndarray:
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
    gt_rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in gt_masks]
    pred_rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in pred_masks]
    return np.asarray(mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=float)


def official_pq(
    gt_masks: Sequence[np.ndarray], pred_masks: Sequence[np.ndarray], threshold: float = 0.50
) -> dict[str, float | int]:
    overlaps = pairwise_iou(gt_masks, pred_masks)
    hits = overlaps > threshold
    tp = int(hits.sum())
    fp = int((hits.sum(axis=0) == 0).sum()) if len(pred_masks) else 0
    fn = int((hits.sum(axis=1) == 0).sum()) if len(gt_masks) else 0
    sum_iou = float(overlaps[hits].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denominator if denominator else 0.0
    sq = sum_iou / tp if tp else 0.0
    rq = tp / denominator if denominator else 0.0
    return {
        "pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn,
        "sum_iou": sum_iou,
    }


def ground_truth_sets(filename: str, indices: Mapping[str, Any]) -> list[list[np.ndarray]]:
    output = []
    for image_id in indices["image_ids_by_file"][filename]:
        info = indices["images_by_id"][image_id]
        output.append(
            [
                annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
                for annotation in indices["annotations_by_image"].get(image_id, [])
                if not annotation.get("iscrowd", 0)
            ]
        )
    return output


def active_folds(cfg: CFG) -> list[int]:
    return [0] if cfg.debug else list(range(cfg.n_folds))


def train_all(
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> None:
    training_started = time.monotonic()
    folds = active_folds(cfg)
    for fold in active_folds(cfg):
        fold_started = time.monotonic()
        print(f"===== TRAIN FOLD {fold} =====")
        yaml_path = prepare_detector_fold(fold, manifest, paths, catalog, indices, cfg, work_dir)
        train_detector(fold, yaml_path, cfg, work_dir)
        gc.collect()
        torch.cuda.empty_cache()
        train_refiner(fold, manifest, paths, catalog, indices, cfg, work_dir)
        gc.collect()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - training_started
        completed = folds.index(fold) + 1
        projected = elapsed / completed * len(folds)
        print(
            {
                "stage": "fold_complete",
                "fold": fold,
                "fold_minutes": round((time.monotonic() - fold_started) / 60, 2),
                "training_elapsed_hours": round(elapsed / 3600, 2),
                "projected_training_hours": round(projected / 3600, 2),
                "runtime_target_hours": cfg.runtime_target_hours,
            }
        )


def run_oof(
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    indices: Mapping[str, Any],
    cfg: CFG,
    work_dir: Path,
) -> pd.DataFrame:
    rows = []
    for fold in active_folds(cfg):
        detector, refiner, device = load_models(fold, cfg, work_dir)
        filenames = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
        if cfg.debug:
            filenames = filenames[:8]
        for number, filename in enumerate(filenames, 1):
            masks, _ = predict_instances(paths["train"] / filename, detector, refiner, device, cfg)
            for annotator_index, gt_masks in enumerate(ground_truth_sets(filename, indices)):
                rows.append(
                    {
                        "filename": filename,
                        "fold": fold,
                        "annotator": annotator_index,
                        "n_gt": len(gt_masks),
                        "n_pred": len(masks),
                        **official_pq(gt_masks, masks, cfg.pq_match_iou),
                    }
                )
            print(f"fold={fold} oof={number}/{len(filenames)} instances={len(masks)}")
        del detector, refiner
        gc.collect()
        torch.cuda.empty_cache()
    annotator_df = pd.DataFrame(rows)
    annotator_df.to_csv(work_dir / "oof_by_annotator.csv", index=False)
    metrics = ["pq", "sq", "rq", "tp", "fp", "fn", "n_gt", "n_pred"]
    observation_df = annotator_df.groupby(["filename", "fold"], as_index=False)[metrics].mean()
    summary = observation_df.groupby("fold")[metrics].mean().reset_index()
    summary = pd.concat(
        [summary, pd.DataFrame([{"fold": "overall", **observation_df[metrics].mean().to_dict()}])],
        ignore_index=True,
    )
    observation_df.to_csv(work_dir / "oof_by_observation.csv", index=False)
    summary.to_csv(work_dir / "oof_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def evaluate_checkpoint_oof(
    family: str,
    paths: Mapping[str, Path],
    indices: Mapping[str, Any],
    cfg: CFG,
    checkpoint_root: Path,
    output_dir: Path,
) -> dict[str, float]:
    """Evaluate one held-out model per image for a saved checkpoint family."""
    manifest_path = checkpoint_root / "fold_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing parent fold manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    rows = []
    for fold in range(cfg.n_folds):
        if family == "spine":
            detector, refiner, device = load_models(fold, cfg, checkpoint_root)
        elif family == "baseline":
            detector, refiner, device = load_legacy_models(fold, cfg, checkpoint_root)
        else:
            raise ValueError(f"Unknown checkpoint family: {family}")
        filenames = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
        for number, filename in enumerate(filenames, 1):
            if family == "spine":
                masks, _ = predict_instances(paths["train"] / filename, detector, refiner, device, cfg)
            else:
                masks, _ = predict_legacy_instances(
                    paths["train"] / filename, detector, refiner, device, cfg
                )
            for annotator_index, gt_masks in enumerate(ground_truth_sets(filename, indices)):
                rows.append(
                    {
                        "filename": filename,
                        "fold": fold,
                        "annotator": annotator_index,
                        "n_gt": len(gt_masks),
                        "n_pred": len(masks),
                        **official_pq(gt_masks, masks, cfg.pq_match_iou),
                    }
                )
            if number % 40 == 0 or number == len(filenames):
                print(
                    {
                        "stage": "oof_recalibration",
                        "family": family,
                        "threshold": cfg.refiner_threshold,
                        "fold": fold,
                        "image": f"{number}/{len(filenames)}",
                    }
                )
        del detector, refiner
        gc.collect()
        torch.cuda.empty_cache()
    annotator_df = pd.DataFrame(rows)
    metrics = ["pq", "sq", "rq", "tp", "fp", "fn", "n_gt", "n_pred"]
    observation_df = annotator_df.groupby(["filename", "fold"], as_index=False)[metrics].mean()
    result = {key: float(value) for key, value in observation_df[metrics].mean().to_dict().items()}
    suffix = f"{family}_t{cfg.refiner_threshold:.2f}".replace(".", "")
    annotator_df.to_csv(output_dir / f"oof_{suffix}_by_annotator.csv", index=False)
    observation_df.to_csv(output_dir / f"oof_{suffix}_by_observation.csv", index=False)
    print({"stage": "oof_recalibration_complete", "family": family, **result})
    return result


def cluster_representative(
    cluster: Sequence[tuple[int, np.ndarray, float]],
) -> tuple[int, np.ndarray, float]:
    """Return the medoid mask, using confidence only to break IoU ties."""
    if len(cluster) == 1:
        return cluster[0]
    agreement = []
    for first in cluster:
        overlaps = [mask_iou(first[1], second[1]) for second in cluster]
        agreement.append(float(np.mean(overlaps)))
    return max(
        zip(agreement, cluster),
        key=lambda value: (value[0], value[1][2]),
    )[1]


def cluster_fold_predictions(
    predictions: Sequence[tuple[int, np.ndarray, float]],
    match_iou: float,
) -> list[list[tuple[int, np.ndarray, float]]]:
    """One-to-one cross-fold matching for thin filament instances.

    Processing one fold at a time and using Hungarian assignment prevents two
    predictions from the same fold from inflating a cluster's vote count.
    """
    by_fold: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for prediction in predictions:
        by_fold.setdefault(prediction[0], []).append(prediction)
    preferred_order = [1, 0, 3, 2, 4]
    fold_order = [fold for fold in preferred_order if fold in by_fold]
    fold_order.extend(sorted(set(by_fold) - set(fold_order)))
    clusters: list[list[tuple[int, np.ndarray, float]]] = []
    for fold in fold_order:
        candidates = sorted(by_fold[fold], key=lambda value: value[2], reverse=True)
        if not clusters:
            clusters = [[candidate] for candidate in candidates]
            continue
        representatives = [cluster_representative(cluster) for cluster in clusters]
        overlaps = np.asarray(
            [
                [mask_iou(representative[1], candidate[1]) for candidate in candidates]
                for representative in representatives
            ],
            dtype=float,
        )
        matched_candidates: set[int] = set()
        if overlaps.size:
            cluster_indices, candidate_indices = linear_sum_assignment(-overlaps)
            for cluster_index, candidate_index in zip(cluster_indices, candidate_indices):
                if overlaps[cluster_index, candidate_index] >= match_iou:
                    clusters[int(cluster_index)].append(candidates[int(candidate_index)])
                    matched_candidates.add(int(candidate_index))
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched_candidates:
                continue
            # A same-fold candidate that overlaps an already claimed cluster
            # is a duplicate, not evidence for an additional instance.
            maximum_overlap = float(overlaps[:, candidate_index].max()) if overlaps.size else 0.0
            if maximum_overlap < match_iou:
                clusters.append([candidate])
    return clusters


def resolve_fused_overlaps(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    supports: Sequence[int],
    policy: FusionPolicy,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Remove overlaps without preserving large duplicate fragments as instances."""
    if not masks:
        return [], [], []
    assigned = np.zeros_like(masks[0], dtype=bool)
    output_masks, output_scores, output_supports = [], [], []
    order = sorted(
        range(len(masks)),
        key=lambda index: (supports[index], scores[index]),
        reverse=True,
    )
    for index in order:
        original = masks[index].astype(bool)
        clean = original & ~assigned
        clean_fraction = clean.sum() / max(original.sum(), 1)
        if clean.sum() < policy.minimum_area or clean_fraction < policy.minimum_clean_fraction:
            continue
        output_masks.append(clean)
        output_scores.append(float(scores[index]))
        output_supports.append(int(supports[index]))
        assigned |= clean
    return output_masks, output_scores, output_supports


def fuse_predictions_with_policy(
    predictions: Sequence[tuple[int, np.ndarray, float]],
    policy: FusionPolicy,
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    clusters = cluster_fold_predictions(predictions, policy.match_iou)
    support_histogram: dict[int, int] = {}
    candidate_masks, candidate_scores, candidate_supports = [], [], []
    for cluster in clusters:
        support = len(cluster)
        support_histogram[support] = support_histogram.get(support, 0) + 1
        if support < policy.minimum_support:
            continue
        if policy.mask_mode == "medoid":
            mask = cluster_representative(cluster)[1].copy()
        elif policy.mask_mode == "majority":
            votes = np.stack([item[1] for item in cluster]).sum(axis=0)
            mask = votes >= (len(cluster) // 2 + 1)
        else:
            raise ValueError(f"Unknown fusion mask mode: {policy.mask_mode}")
        if mask.sum() >= policy.minimum_area:
            candidate_masks.append(mask)
            candidate_scores.append(float(np.mean([item[2] for item in cluster])))
            candidate_supports.append(support)

    used_fallback = False
    if not candidate_masks and clusters:
        fallback = max(
            clusters,
            key=lambda cluster: (len(cluster), np.mean([item[2] for item in cluster])),
        )
        candidate_masks = [cluster_representative(fallback)[1].copy()]
        candidate_scores = [float(np.mean([item[2] for item in fallback]))]
        candidate_supports = [len(fallback)]
        used_fallback = True
    masks, scores, supports = resolve_fused_overlaps(
        candidate_masks, candidate_scores, candidate_supports, policy
    )
    if not masks and candidate_masks:
        best = int(np.argmax(candidate_scores))
        masks, scores, supports = (
            [candidate_masks[best]],
            [candidate_scores[best]],
            [candidate_supports[best]],
        )
        used_fallback = True
    return masks, scores, {
        "clusters": len(clusters),
        "support_histogram": support_histogram,
        "accepted": len(masks),
        "accepted_support": supports,
        "fallback": used_fallback,
    }


def fuse_predictions(
    predictions: Sequence[tuple[int, np.ndarray, float]], cfg: CFG
) -> tuple[list[np.ndarray], list[float]]:
    policy = FusionPolicy(
        "default_s3_medoid",
        match_iou=0.20,
        minimum_support=1 if len(active_folds(cfg)) == 1 else 3,
        mask_mode="medoid",
        minimum_area=max(96, cfg.minimum_area),
    )
    masks, scores, _ = fuse_predictions_with_policy(predictions, policy)
    return masks, scores


def encode_mask(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def run_test(paths: Mapping[str, Path], cfg: CFG, work_dir: Path) -> pd.DataFrame:
    models = [(fold, *load_models(fold, cfg, work_dir)) for fold in active_folds(cfg)]
    rows = []
    test_files = sorted(paths["test"].glob("*.jpeg"))
    if cfg.debug:
        test_files = test_files[:8]
    for number, image_path in enumerate(test_files, 1):
        predictions = []
        for fold, detector, refiner, device in models:
            masks, scores = predict_instances(image_path, detector, refiner, device, cfg)
            predictions.extend((fold, mask, score) for mask, score in zip(masks, scores))
        masks, _ = fuse_predictions(predictions, cfg)
        for instance_index, mask in enumerate(masks, 1):
            rows.append(
                {
                    "filament_id": f"{image_path.stem}_{instance_index}",
                    "segmentation_rle": encode_mask(mask),
                }
            )
        print(f"test={number}/{len(test_files)} instances={len(masks)}")
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    submission.to_csv(work_dir / ("submission_debug.csv" if cfg.debug else "submission.csv"), index=False)
    return submission


def legacy_encoder_parameters(model: LegacySeedRefiner) -> list[nn.Parameter]:
    modules = (model.stem, model.layer1, model.layer2, model.layer3, model.layer4)
    return [parameter for module in modules for parameter in module.parameters()]


def train_pretrained_refiner_strategy(
    strategy: PretrainStrategy,
    fold: int,
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
    backbone_path: Path,
    work_dir: Path,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    strategy_dir = work_dir / strategy.name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = strategy_dir / "refiner.pt"
    train_files = manifest.loc[manifest["fold"] != fold, "filename"].tolist()
    val_files = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
    train_dataset = LegacyRefinerDataset(
        train_files, paths["train"], medoids, indices, cfg, True
    )
    val_dataset = LegacyRefinerDataset(
        val_files, paths["train"], medoids, indices, cfg, False
    )
    gpu_count = torch.cuda.device_count()
    effective_batch = cfg.refiner_batch * max(gpu_count, 1)
    loader_options = {
        "num_workers": min(cfg.num_workers, 4),
        "pin_memory": True,
        "persistent_workers": min(cfg.num_workers, 4) > 0,
    }
    generator = torch.Generator().manual_seed(cfg.seed + fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch * 2,
        shuffle=False,
        **loader_options,
    )
    device = torch.device("cuda:0")
    model = LegacySeedRefiner()
    load_report = load_domain_backbone(model, backbone_path)
    model.to(device)
    encoder_parameters = legacy_encoder_parameters(model)
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    if strategy.encoder_lr == strategy.decoder_lr:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=strategy.encoder_lr, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": strategy.encoder_lr},
                {"params": decoder_parameters, "lr": strategy.decoder_lr},
            ],
            weight_decay=1e-4,
        )
    parallel_model: nn.Module = (
        nn.DataParallel(model, device_ids=list(range(gpu_count))) if gpu_count > 1 else model
    )
    epochs = cfg.refiner_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    seed_everything(cfg.seed + fold)
    print(
        {
            "stage": "pretrained_refiner_train",
            "strategy": asdict(strategy),
            "fold": fold,
            "gpus": gpu_count,
            "effective_batch": effective_batch,
            "train_instances": len(train_dataset),
            "validation_instances": len(val_dataset),
            "backbone": load_report,
        },
        flush=True,
    )
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        encoder_trainable = epoch >= strategy.freeze_encoder_epochs
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_trainable)
        parallel_model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = parallel_model(features)
                loss = (
                    0.45 * F.binary_cross_entropy_with_logits(logits, targets)
                    + 0.55 * soft_dice_loss(logits, targets)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
        parallel_model.eval()
        dice_total = sample_count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).bool()
                predictions = parallel_model(features).sigmoid() >= cfg.refiner_threshold
                intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
                denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
                dice_total += (
                    (2 * intersection + 1e-7) / (denominator + 1e-7)
                ).sum().item()
                sample_count += len(features)
        validation_dice = dice_total / max(sample_count, 1)
        row = {
            "strategy": strategy.name,
            "epoch": epoch + 1,
            "encoder_trainable": encoder_trainable,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_dice": validation_dice,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[-1]["lr"],
            "epoch_minutes": (time.monotonic() - epoch_started) / 60,
        }
        history.append(row)
        print(json.dumps({"event": "refiner_epoch", **row}, separators=(",", ":")), flush=True)
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(strategy_dir / "history.csv", index=False)
    del parallel_model, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint, history_frame, load_report


def evaluate_pretrained_refiner_oof(
    strategy: str,
    checkpoint: Path,
    detector_root: Path,
    fold: int,
    manifest: pd.DataFrame,
    paths: Mapping[str, Path],
    indices: Mapping[str, Any],
    cfg: CFG,
    output_dir: Path,
) -> dict[str, float]:
    from ultralytics import YOLO

    device = torch.device("cuda:0")
    detector = YOLO(str(detector_root / "models" / f"fold_{fold}" / "detector.pt"))
    refiner = LegacySeedRefiner()
    refiner.load_state_dict(torch.load(checkpoint, map_location=device))
    refiner.to(device).eval()
    rows = []
    filenames = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
    for number, filename in enumerate(filenames, 1):
        masks, _ = predict_legacy_instances(
            paths["train"] / filename, detector, refiner, device, cfg
        )
        for annotator, gt_masks in enumerate(ground_truth_sets(filename, indices)):
            rows.append(
                {
                    "strategy": strategy,
                    "filename": filename,
                    "fold": fold,
                    "annotator": annotator,
                    "n_gt": len(gt_masks),
                    "n_pred": len(masks),
                    **official_pq(gt_masks, masks, cfg.pq_match_iou),
                }
            )
        if number % 20 == 0 or number == len(filenames):
            print(
                {
                    "stage": "pretrained_refiner_oof",
                    "strategy": strategy,
                    "image": f"{number}/{len(filenames)}",
                },
                flush=True,
            )
    annotator_df = pd.DataFrame(rows)
    metric_columns = [
        "pq", "sq", "rq", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred"
    ]
    observation_df = annotator_df.groupby(
        ["strategy", "filename", "fold"], as_index=False
    )[metric_columns].mean()
    annotator_df.to_csv(output_dir / strategy / "oof_by_annotator.csv", index=False)
    observation_df.to_csv(output_dir / strategy / "oof_by_observation.csv", index=False)
    result = {
        key: float(value)
        for key, value in observation_df[metric_columns].mean().to_dict().items()
    }
    totals = annotator_df[["sum_iou", "tp", "fp", "fn"]].sum()
    result["official_pq"] = float(
        totals["sum_iou"]
        / (totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"])
    )
    del detector, refiner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_pretrained_refiner_ablation(
    paths: Mapping[str, Path],
    detector_root: Path,
    backbone_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Train the three requested one-fold strategies and select on true OOF PQ."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(detector_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.refiner_epochs = 8
    cfg.num_workers = 4
    fold = 0
    results: dict[str, Any] = {}
    for strategy in PRETRAIN_STRATEGIES:
        checkpoint, history, load_report = train_pretrained_refiner_strategy(
            strategy, fold, manifest, paths, medoids, indices, cfg,
            backbone_path, work_dir,
        )
        oof = evaluate_pretrained_refiner_oof(
            strategy.name, checkpoint, detector_root, fold, manifest,
            paths, indices, cfg, work_dir,
        )
        results[strategy.name] = {
            "strategy": asdict(strategy),
            "checkpoint": str(checkpoint),
            "backbone_load": load_report,
            "best_crop_dice": float(history["validation_dice"].max()),
            "oof": oof,
        }
        print(
            json.dumps(
                {"event": "strategy_complete", "strategy": strategy.name, **results[strategy.name]},
                separators=(",", ":"), default=str,
            ),
            flush=True,
        )
    baseline_oof_path = detector_root / "oof_by_observation.csv"
    baseline_oof = pd.read_csv(baseline_oof_path)
    baseline_fold = baseline_oof.loc[baseline_oof["fold"] == fold]
    metric_columns = [
        "pq", "sq", "rq", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred"
    ]
    baseline = {
        key: float(value)
        for key, value in baseline_fold[metric_columns].mean().to_dict().items()
    }
    baseline_annotator = pd.read_csv(detector_root / "oof_by_annotator.csv")
    baseline_annotator = baseline_annotator.loc[baseline_annotator["fold"] == fold]
    baseline_totals = baseline_annotator[["sum_iou", "tp", "fp", "fn"]].sum()
    baseline["official_pq"] = float(
        baseline_totals["sum_iou"]
        / (
            baseline_totals["tp"]
            + 0.5 * baseline_totals["fp"]
            + 0.5 * baseline_totals["fn"]
        )
    )
    best_name = max(results, key=lambda name: results[name]["oof"]["official_pq"])
    report = {
        "status": "complete",
        "fold": fold,
        "backbone": str(backbone_path),
        "baseline_fold_oof": baseline,
        "strategies": results,
        "best_strategy": best_name,
        "best_official_pq_delta": (
            results[best_name]["oof"]["official_pq"] - baseline["official_pq"]
        ),
        "best_macro_pq_delta": results[best_name]["oof"]["pq"] - baseline["pq"],
        "promotion_rule": "Train five folds only if OOF PQ improves without an RQ collapse.",
    }
    (work_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "ablation_complete", **report}, separators=(",", ":")), flush=True)
    return report


@torch.no_grad()
def raw_legacy_candidates(
    image_path: Path,
    detector: Any,
    device: torch.device,
    cfg: CFG,
    confidence: float = 0.05,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Run the detector once at high recall for all audit configurations."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    result = detector.predict(
        str(image_path),
        imgsz=cfg.detector_size,
        conf=confidence,
        iou=0.80,
        agnostic_nms=True,
        retina_masks=True,
        device=device.index if device.type == "cuda" else "cpu",
        verbose=False,
    )[0]
    if result.masks is None:
        return image, [], []
    seeds = [mask.cpu().numpy().astype(bool) for mask in result.masks.data]
    if seeds and seeds[0].shape != image.shape:
        seeds = [
            cv2.resize(seed.astype(np.uint8), image.shape[::-1], interpolation=cv2.INTER_NEAREST)
            .astype(bool)
            for seed in seeds
        ]
    scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    return image, seeds, scores


@torch.no_grad()
def legacy_probability_record(
    image: np.ndarray,
    seed_mask: np.ndarray,
    detector_score: float,
    model: LegacySeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> dict[str, Any]:
    return legacy_probability_records(
        image, [seed_mask], [detector_score], model, device, cfg
    )[0]


@torch.no_grad()
def legacy_probability_records(
    image: np.ndarray,
    seed_masks: Sequence[np.ndarray],
    detector_scores: Sequence[float],
    model: LegacySeedRefiner,
    device: torch.device,
    cfg: CFG,
) -> list[dict[str, Any]]:
    """Batch all seed crops and four TTA views while retaining identical outputs."""
    if not seed_masks:
        return []
    size = (cfg.refiner_crop, cfg.refiner_crop)
    bounds: list[tuple[int, int, int, int]] = []
    feature_arrays = []
    for seed_mask in seed_masks:
        x0, y0, x1, y1 = square_bounds(
            seed_mask, cfg.refiner_context, cfg.refiner_min_crop
        )
        bounds.append((x0, y0, x1, y1))
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed_mask[y0:y1, x0:x1].astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        feature_arrays.append(
            np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32)
            / 255.0
        )
    tensor = torch.from_numpy(np.stack(feature_arrays)).to(device)
    transforms = [
        lambda value: value,
        lambda value: value.flip(-1),
        lambda value: value.flip(-2),
        lambda value: value.flip((-2, -1)),
    ]
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    probability_chunks = []
    for chunk in transformed.split(16):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probability_chunks.append(model(chunk).sigmoid().float().cpu())
    probabilities = torch.cat(probability_chunks, dim=0)
    count = len(seed_masks)
    records = []
    for index, (seed_mask, detector_score, crop_bounds) in enumerate(
        zip(seed_masks, detector_scores, bounds)
    ):
        views = [
            transforms[view](probabilities[view * count + index : view * count + index + 1])[
                0, 0
            ].numpy()
            for view in range(4)
        ]
        x0, y0, x1, y1 = crop_bounds
        records.append(
            {
                "seed": seed_mask,
                "detector_score": float(detector_score),
                "bounds": crop_bounds,
                "base_probability": cv2.resize(
                    views[0], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
                ),
                "tta_probability": cv2.resize(
                    np.mean(views, axis=0),
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR,
                ),
            }
        )
    return records


def audit_candidate_indices(
    records: Sequence[Mapping[str, Any]], setting: AuditSetting
) -> list[int]:
    eligible = [
        index
        for index, record in enumerate(records)
        if float(record["detector_score"]) >= setting.detector_conf
    ]
    # Reproduce the baseline's low-confidence second pass instead of emitting no mask.
    if not eligible:
        eligible = list(range(len(records)))
    kept: list[int] = []
    for index in sorted(
        eligible, key=lambda item: float(records[item]["detector_score"]), reverse=True
    ):
        seed = records[index]["seed"]
        if all(mask_iou(seed, records[other]["seed"]) < setting.seed_nms_iou for other in kept):
            kept.append(index)
    return kept


def apply_audit_morphology(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask.astype(bool)
    operations = {
        "open3": ((cv2.MORPH_OPEN, 3),),
        "close3": ((cv2.MORPH_CLOSE, 3),),
        "close5": ((cv2.MORPH_CLOSE, 5),),
        "open3_close3": ((cv2.MORPH_OPEN, 3), (cv2.MORPH_CLOSE, 3)),
    }
    if mode not in operations:
        raise ValueError(f"Unknown morphology mode: {mode}")
    output = mask.astype(np.uint8)
    for operation, width in operations[mode]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        output = cv2.morphologyEx(output, operation, kernel)
    return output.astype(bool)


def threshold_probability_record(
    record: Mapping[str, Any], setting: AuditSetting
) -> tuple[np.ndarray, float]:
    base_probability = record["base_probability"]
    tta_probability = record["tta_probability"]
    base_mask = base_probability >= setting.refiner_threshold
    candidate_mask = tta_probability >= setting.refiner_threshold
    use_candidate = True
    if base_mask.any():
        area_ratio = candidate_mask.sum() / max(base_mask.sum(), 1)
        use_candidate = (
            0.60 <= area_ratio <= 1.70
            and binary_dice(base_mask, candidate_mask) >= 0.62
            and component_count(candidate_mask) <= component_count(base_mask) + 1
        )
    probability = tta_probability if use_candidate else base_probability
    crop_mask = apply_audit_morphology(
        probability >= setting.refiner_threshold, setting.morphology
    )
    x0, y0, x1, y1 = record["bounds"]
    full_mask = np.zeros_like(record["seed"], dtype=bool)
    full_mask[y0:y1, x0:x1] = crop_mask
    if not full_mask.any():
        full_mask = record["seed"].astype(bool)
    confidence = float(probability[crop_mask].mean()) if crop_mask.any() else 0.0
    return full_mask, confidence


def predictions_for_setting(
    records: Sequence[Mapping[str, Any]], setting: AuditSetting
) -> tuple[list[np.ndarray], list[float]]:
    masks, scores = [], []
    for index in audit_candidate_indices(records, setting):
        mask, quality = threshold_probability_record(records[index], setting)
        if mask.sum() < setting.minimum_area:
            continue
        masks.append(mask)
        scores.append(float(records[index]["detector_score"]) * quality)
    if not masks:
        return [], []
    return resolve_overlaps(masks, scores, setting.minimum_area)


def predictions_from_oracle_records(
    records: Sequence[Mapping[str, Any]], setting: AuditSetting
) -> list[np.ndarray]:
    masks, scores = [], []
    for record in records:
        mask, quality = threshold_probability_record(record, setting)
        if mask.sum() >= setting.minimum_area:
            masks.append(mask)
            scores.append(max(quality, 1e-6))
    return resolve_overlaps(masks, scores, setting.minimum_area)[0] if masks else []


def failure_taxonomy(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    overlaps: np.ndarray | None = None,
) -> dict[str, int]:
    if overlaps is None:
        overlaps = pairwise_iou(gt_masks, pred_masks)
    hits = overlaps > 0.50
    output = {
        "fragmented_gt": 0,
        "merged_pred": 0,
        "boundary_fn": 0,
        "weak_fn": 0,
        "complete_miss": 0,
        "boundary_fp": 0,
        "weak_fp": 0,
        "hallucinated_fp": 0,
    }
    for gt_index in range(len(gt_masks)):
        if hits[gt_index].any():
            continue
        values = overlaps[gt_index] if len(pred_masks) else np.zeros(0)
        maximum = float(values.max()) if values.size else 0.0
        if int((values > 0.05).sum()) >= 2:
            output["fragmented_gt"] += 1
        elif maximum >= 0.30:
            output["boundary_fn"] += 1
        elif maximum >= 0.10:
            output["weak_fn"] += 1
        else:
            output["complete_miss"] += 1
    for pred_index in range(len(pred_masks)):
        if hits[:, pred_index].any():
            continue
        values = overlaps[:, pred_index] if len(gt_masks) else np.zeros(0)
        maximum = float(values.max()) if values.size else 0.0
        if int((values > 0.05).sum()) >= 2:
            output["merged_pred"] += 1
        elif maximum >= 0.30:
            output["boundary_fp"] += 1
        elif maximum >= 0.10:
            output["weak_fp"] += 1
        else:
            output["hallucinated_fp"] += 1
    return output


def audit_metric_row(
    filename: str,
    fold: int,
    annotator: int,
    variant: str,
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    gt_rles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if gt_rles is None:
        overlaps = pairwise_iou(gt_masks, pred_masks)
    elif not gt_rles or not pred_masks:
        overlaps = np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
    else:
        pred_rles = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            for mask in pred_masks
        ]
        overlaps = np.asarray(
            mask_utils.iou(list(gt_rles), pred_rles, [0] * len(pred_rles)), dtype=float
        )
    hits = overlaps > 0.50
    tp = int(hits.sum())
    fp = int((hits.sum(axis=0) == 0).sum()) if len(pred_masks) else 0
    fn = int((hits.sum(axis=1) == 0).sum()) if len(gt_masks) else 0
    sum_iou = float(overlaps[hits].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / denominator if denominator else 0.0
    return {
        "filename": filename,
        "fold": fold,
        "annotator": annotator,
        "variant": variant,
        "n_gt": len(gt_masks),
        "n_pred": len(pred_masks),
        "pq": pq,
        "sq": sum_iou / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
        **failure_taxonomy(gt_masks, pred_masks, overlaps),
    }


def medoid_masks_for_file(
    filename: str,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
) -> list[np.ndarray]:
    image_id = medoids[filename]
    info = indices["images_by_id"][image_id]
    return [
        annotation_to_mask(annotation, int(info["height"]), int(info["width"]))
        for annotation in indices["annotations_by_image"].get(image_id, [])
        if not annotation.get("iscrowd", 0)
    ]


def box_seed(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs):
        output[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return output


def detector_recovery_oracle(
    gt_masks: Sequence[np.ndarray], seeds: Sequence[np.ndarray], minimum_iou: float
) -> list[np.ndarray]:
    overlaps = pairwise_iou(gt_masks, seeds)
    if not overlaps.size:
        return []
    gt_indices, seed_indices = linear_sum_assignment(-overlaps)
    return [
        gt_masks[int(gt_index)].copy()
        for gt_index, seed_index in zip(gt_indices, seed_indices)
        if overlaps[int(gt_index), int(seed_index)] >= minimum_iou
    ]


def baseline_audit_worker(
    gpu: int,
    folds: Sequence[int],
    checkpoint_root_text: str,
    output_dir_text: str,
) -> None:
    torch.cuda.set_device(gpu)
    seed_everything(20260815 + gpu)
    paths = discover_paths()
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    checkpoint_root = Path(checkpoint_root_text)
    manifest = pd.read_csv(checkpoint_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    base_setting = AuditSetting("baseline")
    rows: list[dict[str, Any]] = []
    for fold in folds:
        detector, refiner, device = load_legacy_models(fold, cfg, checkpoint_root)
        filenames = manifest.loc[manifest["fold"] == fold, "filename"].tolist()
        for number, filename in enumerate(filenames, 1):
            image, seeds, detector_scores = raw_legacy_candidates(
                paths["train"] / filename, detector, device, cfg
            )
            if not seeds:
                seeds = [classical_fallback_seed(image, cfg.minimum_area)]
                detector_scores = [0.01]
            # No tested setting uses conf < 0.10. Preserve all candidates only
            # for the baseline's genuine low-confidence fallback case.
            if any(score >= 0.20 for score in detector_scores):
                refinement_indices = [
                    index for index, score in enumerate(detector_scores) if score >= 0.10
                ]
            else:
                refinement_indices = list(range(len(seeds)))
            refinement_seeds = [seeds[index] for index in refinement_indices]
            refinement_scores = [detector_scores[index] for index in refinement_indices]
            records = legacy_probability_records(
                image, refinement_seeds, refinement_scores, refiner, device, cfg
            )
            gt_sets = ground_truth_sets(filename, indices)
            gt_rle_sets = [
                [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in gt_masks]
                for gt_masks in gt_sets
            ]
            setting_predictions = {
                setting.name: predictions_for_setting(records, setting)[0]
                for setting in audit_settings()
            }
            detector_indices = audit_candidate_indices(records, base_setting)
            detector_masks, detector_mask_scores = resolve_overlaps(
                [records[index]["seed"] for index in detector_indices],
                [float(records[index]["detector_score"]) for index in detector_indices],
                base_setting.minimum_area,
            )
            medoid_masks = medoid_masks_for_file(filename, medoids, indices)
            gt_seed_records = legacy_probability_records(
                image, medoid_masks, [1.0] * len(medoid_masks), refiner, device, cfg
            )
            medoid_boxes = [box_seed(mask) for mask in medoid_masks]
            box_records = legacy_probability_records(
                image, medoid_boxes, [1.0] * len(medoid_boxes), refiner, device, cfg
            )
            oracle_predictions = {
                "detector_masks": detector_masks,
                "medoid_annotation": medoid_masks,
                "gt_seed_refiner": predictions_from_oracle_records(gt_seed_records, base_setting),
                "gt_box_refiner": predictions_from_oracle_records(box_records, base_setting),
            }
            for annotator, gt_masks in enumerate(gt_sets):
                for variant, masks in setting_predictions.items():
                    rows.append(
                        audit_metric_row(
                            filename, fold, annotator, variant, gt_masks, masks,
                            gt_rle_sets[annotator],
                        )
                    )
                for variant, masks in oracle_predictions.items():
                    rows.append(
                        audit_metric_row(
                            filename, fold, annotator, variant, gt_masks, masks,
                            gt_rle_sets[annotator],
                        )
                    )
                for minimum_iou in (0.01, 0.05, 0.10, 0.20, 0.30):
                    variant = f"detector_oracle_iou_{minimum_iou:.2f}"
                    masks = detector_recovery_oracle(gt_masks, seeds, minimum_iou)
                    rows.append(
                        audit_metric_row(
                            filename, fold, annotator, variant, gt_masks, masks,
                            gt_rle_sets[annotator],
                        )
                    )
            if number % 20 == 0 or number == len(filenames):
                print(
                    {
                        "stage": "baseline_audit",
                        "gpu": gpu,
                        "fold": fold,
                        "image": f"{number}/{len(filenames)}",
                        "raw_candidates": len(seeds),
                        "refined_candidates": len(records),
                    },
                    flush=True,
                )
        del detector, refiner
        gc.collect()
        torch.cuda.empty_cache()
    output_dir = Path(output_dir_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"audit_gpu_{gpu}.csv", index=False)


def run_baseline_audit(
    checkpoint_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Parallel OOF diagnosis, post-processing sweep, and useful oracle bounds."""
    work_dir.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("The baseline audit requires a CUDA device")
    worker_count = min(gpu_count, 2)
    context = mp.get_context("spawn")
    processes = []
    for gpu in range(worker_count):
        folds = [fold for fold in range(5) if fold % worker_count == gpu]
        process = context.Process(
            target=baseline_audit_worker,
            args=(gpu, folds, str(checkpoint_root), str(work_dir)),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Audit worker failed with exit code {process.exitcode}")

    annotator_df = pd.concat(
        [pd.read_csv(work_dir / f"audit_gpu_{gpu}.csv") for gpu in range(worker_count)],
        ignore_index=True,
    )
    metric_columns = [
        "pq", "sq", "rq", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred",
        "fragmented_gt", "merged_pred", "boundary_fn", "weak_fn", "complete_miss",
        "boundary_fp", "weak_fp", "hallucinated_fp",
    ]
    observation_df = annotator_df.groupby(
        ["variant", "filename", "fold"], as_index=False
    )[metric_columns].mean()
    summary = observation_df.groupby("variant", as_index=False)[metric_columns].mean()
    summary["observations"] = observation_df.groupby("variant")["filename"].count().values
    micro_counts = annotator_df.groupby("variant", as_index=False)[
        ["sum_iou", "tp", "fp", "fn"]
    ].sum()
    micro_counts["official_pq"] = micro_counts["sum_iou"] / (
        micro_counts["tp"] + 0.5 * micro_counts["fp"] + 0.5 * micro_counts["fn"]
    )
    summary = summary.merge(
        micro_counts[["variant", "official_pq"]], on="variant", how="left", validate="one_to_one"
    )
    setting_names = {setting.name for setting in audit_settings()}
    candidates = summary.loc[summary["variant"].isin(setting_names)]
    best_row = candidates.sort_values(
        ["official_pq", "rq", "sq"], ascending=False
    ).iloc[0]
    best_variant = str(best_row["variant"])
    selected = {"baseline", best_variant}
    fold_summary = (
        observation_df.loc[observation_df["variant"].isin(selected)]
        .groupby(["variant", "fold"], as_index=False)[metric_columns]
        .mean()
    )
    annotator_df.to_csv(work_dir / "audit_by_annotator.csv", index=False)
    observation_df.to_csv(work_dir / "audit_by_observation.csv", index=False)
    summary.sort_values("official_pq", ascending=False).to_csv(
        work_dir / "audit_summary.csv", index=False
    )
    fold_summary.to_csv(work_dir / "audit_fold_summary.csv", index=False)
    baseline_row = summary.loc[summary["variant"] == "baseline"].iloc[0]
    report = {
        "status": "complete",
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "observations": int(observation_df["filename"].nunique()),
        "settings_tested": len(setting_names),
        "baseline": {
            **{key: float(baseline_row[key]) for key in metric_columns},
            "official_pq": float(baseline_row["official_pq"]),
        },
        "best_postprocess": {
            "variant": best_variant,
            "setting": asdict(next(setting for setting in audit_settings() if setting.name == best_variant)),
            **{key: float(best_row[key]) for key in metric_columns},
            "official_pq": float(best_row["official_pq"]),
        },
        "official_pq_delta": float(
            best_row["official_pq"] - baseline_row["official_pq"]
        ),
        "oracle_pq": {
            row.variant: float(row.pq)
            for row in summary.itertuples()
            if "oracle" in row.variant or row.variant in {"detector_masks", "medoid_annotation", "gt_seed_refiner", "gt_box_refiner"}
        },
        "files": {
            "summary": str(work_dir / "audit_summary.csv"),
            "fold_summary": str(work_dir / "audit_fold_summary.csv"),
            "observations": str(work_dir / "audit_by_observation.csv"),
        },
    }
    (work_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "audit_complete", **report}, separators=(",", ":")), flush=True)
    return report


def run_recalibration(
    paths: Mapping[str, Path],
    cfg: CFG,
    checkpoint_roots: Mapping[str, Path],
    work_dir: Path,
) -> dict[str, Any]:
    """Re-run only test inference and write conservative fusion variants."""
    required = {"spine", "baseline"}
    missing = required - set(checkpoint_roots)
    if missing:
        raise FileNotFoundError(f"Missing parent checkpoint families: {sorted(missing)}")
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg = legacy_inference_cfg()
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    metric_columns = ["pq", "sq", "rq", "tp", "fp", "fn", "n_gt", "n_pred"]
    saved_summary = pd.read_csv(checkpoint_roots["spine"] / "oof_summary.csv")
    saved_overall = saved_summary.loc[saved_summary["fold"].astype(str) == "overall"].iloc[0]
    spine_calibration = {
        "0.58": {column: float(saved_overall[column]) for column in metric_columns}
    }
    for threshold in (0.62, 0.65):
        threshold_cfg = replace(cfg, refiner_threshold=threshold)
        spine_calibration[f"{threshold:.2f}"] = evaluate_checkpoint_oof(
            "spine",
            paths,
            indices,
            threshold_cfg,
            checkpoint_roots["spine"],
            work_dir,
        )
    best_spine_threshold = float(
        max(
            spine_calibration,
            key=lambda threshold: (
                spine_calibration[threshold]["pq"],
                -spine_calibration[threshold]["n_pred"],
            ),
        )
    )
    spine_cfg = replace(
        cfg,
        refiner_threshold=best_spine_threshold,
    )
    baseline_oof = evaluate_checkpoint_oof(
        "baseline",
        paths,
        indices,
        baseline_cfg,
        checkpoint_roots["baseline"],
        work_dir,
    )
    print(
        {
            "stage": "threshold_selection",
            "spine": spine_calibration,
            "selected_spine_threshold": best_spine_threshold,
            "baseline": baseline_oof,
        }
    )
    family_models = {
        "spine": [
            (fold, *load_models(fold, spine_cfg, checkpoint_roots["spine"]))
            for fold in range(spine_cfg.n_folds)
        ],
        "baseline": [
            (fold, *load_legacy_models(fold, baseline_cfg, checkpoint_roots["baseline"]))
            for fold in range(baseline_cfg.n_folds)
        ],
    }
    rows_by_variant: dict[str, list[dict[str, str]]] = {
        f"{family}_{policy.name}": []
        for family in family_models
        for policy in RECALIBRATION_POLICIES
    }
    diagnostic_rows: list[dict[str, Any]] = []
    test_files = sorted(paths["test"].glob("*.jpeg"))
    for number, image_path in enumerate(test_files, 1):
        progress: dict[str, int] = {}
        for family, models in family_models.items():
            family_cfg = spine_cfg if family == "spine" else baseline_cfg
            predictions = []
            for fold, detector, refiner, device in models:
                if family == "spine":
                    masks, scores = predict_instances(
                        image_path, detector, refiner, device, family_cfg
                    )
                else:
                    masks, scores = predict_legacy_instances(
                        image_path, detector, refiner, device, family_cfg
                    )
                predictions.extend(
                    (fold, mask, score) for mask, score in zip(masks, scores)
                )
            for policy in RECALIBRATION_POLICIES:
                variant = f"{family}_{policy.name}"
                masks, _, diagnostics = fuse_predictions_with_policy(predictions, policy)
                progress[variant] = len(masks)
                for instance_index, mask in enumerate(masks, 1):
                    rows_by_variant[variant].append(
                        {
                            "filament_id": f"{image_path.stem}_{instance_index}",
                            "segmentation_rle": encode_mask(mask),
                        }
                    )
                diagnostic_rows.append(
                    {
                        "filename": image_path.name,
                        "family": family,
                        "policy": policy.name,
                        "raw_instances": len(predictions),
                        "clusters": diagnostics["clusters"],
                        "instances": len(masks),
                        "mean_area": float(np.mean([mask.sum() for mask in masks])),
                        "fallback": diagnostics["fallback"],
                        "support_histogram": json.dumps(
                            diagnostics["support_histogram"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
        if number % 10 == 0 or number == len(test_files):
            print(
                {
                    "stage": "recalibration",
                    "image": f"{number}/{len(test_files)}",
                    "outputs": progress,
                }
            )

    test_ids = {path.stem for path in test_files}
    summaries: dict[str, Any] = {}
    for variant, rows in rows_by_variant.items():
        submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        submitted_ids = set(submission["filament_id"].str.rsplit("_", n=1).str[0])
        if submitted_ids != test_ids or not submission["filament_id"].is_unique:
            raise RuntimeError(
                f"Invalid {variant}: missing={len(test_ids - submitted_ids)}, "
                f"duplicate_ids={submission['filament_id'].duplicated().sum()}"
            )
        output_path = work_dir / f"submission_{variant}.csv"
        submission.to_csv(output_path, index=False)
        areas = [row for row in diagnostic_rows if f"{row['family']}_{row['policy']}" == variant]
        summaries[variant] = {
            "path": str(output_path),
            "rows": len(submission),
            "images": len(submitted_ids),
            "mean_instances": len(submission) / len(submitted_ids),
            "mean_mask_area": float(np.mean([row["mean_area"] for row in areas])),
            "fallback_images": int(sum(bool(row["fallback"]) for row in areas)),
        }
    pd.DataFrame(diagnostic_rows).to_csv(work_dir / "recalibration_diagnostics.csv", index=False)
    report = {
        "mode": "inference-only-recalibration",
        "checkpoint_roots": {key: str(value) for key, value in checkpoint_roots.items()},
        "oof": {
            "spine_by_threshold": spine_calibration,
            "selected_spine_threshold": best_spine_threshold,
            "baseline": baseline_oof,
        },
        "spine_inference": {
            "refiner_threshold": spine_cfg.refiner_threshold,
            "minimum_area": spine_cfg.minimum_area,
        },
        "policies": [asdict(policy) for policy in RECALIBRATION_POLICIES],
        "variants": summaries,
    }
    (work_dir / "recalibration_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report)
    return report


class ProposalRefinerDataset(Dataset):
    """Actual detector proposals paired with one-to-one medoid targets."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        image_dir: Path,
        cfg: CFG,
        augment: bool,
    ) -> None:
        self.samples = list(samples)
        self.image_dir = image_dir
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(
            str(self.image_dir / str(sample["filename"])), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise FileNotFoundError(self.image_dir / str(sample["filename"]))
        seed = mask_utils.decode(sample["seed_rle"]).astype(bool)
        keep = float(sample["keep"])
        target = (
            mask_utils.decode(sample["target_rle"]).astype(np.float32)
            if keep
            else np.zeros_like(seed, dtype=np.float32)
        )
        x0, y0, x1, y1 = square_bounds(
            seed, self.cfg.refiner_context, self.cfg.refiner_min_crop
        )
        size = (self.cfg.refiner_crop,) * 2
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1].astype(np.uint8), size,
            interpolation=cv2.INTER_NEAREST,
        )
        # A soft resized target avoids teaching staircase boundaries at crop scale.
        target_crop = cv2.resize(
            target[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(
            np.float32
        ) / 255.0
        if self.augment:
            rotation = random.randrange(4)
            features = np.rot90(features, rotation, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, rotation).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target_crop = target_crop[:, ::-1].copy()
            if random.random() < 0.5:
                features = features[:, ::-1, :].copy()
                target_crop = target_crop[::-1, :].copy()
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.04, 0.04)
            features[:2] = np.clip(features[:2] * gain + bias, 0, 1)
        return (
            torch.from_numpy(features),
            torch.from_numpy(target_crop[None].astype(np.float32)),
            torch.tensor(keep, dtype=torch.float32),
        )


class ProposalRefinerV2(LegacySeedRefiner):
    """Baseline refiner plus a proposal keep/reject head."""

    def __init__(self) -> None:
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        quality = self.quality_head(x4).squeeze(1)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        return self.head(self.up0(x)), quality


def binary_rle(mask: np.ndarray) -> dict[str, Any]:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def selected_proposal_indices(
    seeds: Sequence[np.ndarray], scores: Sequence[float], cfg: CFG
) -> list[int]:
    if not seeds:
        return []
    records = [
        {"seed": seed, "detector_score": float(score)}
        for seed, score in zip(seeds, scores)
    ]
    setting = AuditSetting(
        "proposal_v2",
        detector_conf=cfg.detector_conf,
        seed_nms_iou=cfg.seed_nms_iou,
        refiner_threshold=cfg.refiner_threshold,
        minimum_area=cfg.minimum_area,
    )
    return audit_candidate_indices(records, setting)[:20]


@torch.no_grad()
def build_proposal_catalog(
    filenames: Sequence[str],
    image_dir: Path,
    detector: Any,
    device: torch.device,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
    fold: int,
) -> list[dict[str, Any]]:
    """Create proposal samples once; only one-to-one candidates are positives."""
    samples: list[dict[str, Any]] = []
    positive_count = negative_count = 0
    for number, filename in enumerate(filenames, 1):
        image, seeds, scores = raw_legacy_candidates(
            image_dir / filename, detector, device, cfg
        )
        if not seeds:
            seeds = [classical_fallback_seed(image, cfg.minimum_area)]
            scores = [0.01]
        selected = selected_proposal_indices(seeds, scores, cfg)
        chosen_seeds = [seeds[index] for index in selected]
        chosen_scores = [float(scores[index]) for index in selected]
        targets = medoid_masks_for_file(filename, medoids, indices)
        overlaps = pairwise_iou(targets, chosen_seeds)
        assigned: dict[int, tuple[int, float]] = {}
        if overlaps.size:
            target_indices, seed_indices = linear_sum_assignment(-overlaps)
            for target_index, seed_index in zip(target_indices, seed_indices):
                overlap = float(overlaps[int(target_index), int(seed_index)])
                if overlap >= 0.01:
                    assigned[int(seed_index)] = (int(target_index), overlap)
        for seed_index, (seed, detector_score) in enumerate(
            zip(chosen_seeds, chosen_scores)
        ):
            match = assigned.get(seed_index)
            keep = float(match is not None)
            target = targets[match[0]] if match is not None else None
            samples.append(
                {
                    "filename": filename,
                    "fold": fold,
                    "seed_rle": binary_rle(seed),
                    "target_rle": binary_rle(target) if target is not None else None,
                    "detector_score": detector_score,
                    "seed_target_iou": match[1] if match is not None else 0.0,
                    "keep": keep,
                }
            )
            positive_count += int(keep)
            negative_count += int(not keep)
        if number % 40 == 0 or number == len(filenames):
            print(
                {
                    "stage": "proposal_catalog",
                    "fold": fold,
                    "image": f"{number}/{len(filenames)}",
                    "samples": len(samples),
                    "positive": positive_count,
                    "negative": negative_count,
                },
                flush=True,
            )
    return samples


def proposal_mask_losses(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dilation = F.max_pool2d(targets, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-targets, 3, stride=1, padding=1)
    boundary = (dilation - erosion).clamp(0, 1)
    pixel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    boundary_bce = (pixel_bce * (1.0 + 2.5 * boundary)).mean(dim=(1, 2, 3))
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return boundary_bce, dice


def initialize_proposal_refiner(
    checkpoint: Path, device: torch.device
) -> ProposalRefinerV2:
    model = ProposalRefinerV2()
    result = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    expected_missing = {
        "quality_head.2.weight", "quality_head.2.bias",
        "quality_head.5.weight", "quality_head.5.bias",
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected v2 initialization: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model.to(device)


def train_proposal_refiner(
    fold: int,
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    baseline_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    fold_dir = output_dir / "models" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "proposal_refiner_v2.pt"
    train_dataset = ProposalRefinerDataset(train_samples, image_dir, cfg, True)
    validation_dataset = ProposalRefinerDataset(
        validation_samples, image_dir, cfg, False
    )
    gpu_count = torch.cuda.device_count()
    effective_batch = cfg.refiner_batch * max(gpu_count, 1)
    loader_options = {
        "num_workers": min(cfg.num_workers, 4),
        "pin_memory": True,
        "persistent_workers": min(cfg.num_workers, 4) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + fold),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=effective_batch * 2,
        shuffle=False,
        **loader_options,
    )
    device = torch.device("cuda:0")
    model = initialize_proposal_refiner(baseline_checkpoint, device)
    parallel_model: nn.Module = (
        nn.DataParallel(model, device_ids=list(range(gpu_count)))
        if gpu_count > 1 else model
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = cfg.refiner_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_validation_score = -1.0
    history: list[dict[str, Any]] = []
    print(
        {
            "stage": "proposal_refiner_train",
            "fold": fold,
            "gpus": gpu_count,
            "effective_batch": effective_batch,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "train_positive": int(sum(float(x["keep"]) for x in train_samples)),
            "validation_positive": int(
                sum(float(x["keep"]) for x in validation_samples)
            ),
        },
        flush=True,
    )
    for epoch in range(epochs):
        started = time.monotonic()
        parallel_model.train()
        loss_sum = 0.0
        for features, targets, keep_targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            keep_targets = keep_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                mask_logits, quality_logits = parallel_model(features)
                boundary_bce, dice = proposal_mask_losses(mask_logits, targets)
                positives = keep_targets
                negatives = 1.0 - keep_targets
                positive_loss = (
                    ((0.45 * boundary_bce + 0.55 * dice) * positives).sum()
                    / positives.sum().clamp_min(1.0)
                )
                negative_loss = (
                    (boundary_bce * negatives).sum() / negatives.sum().clamp_min(1.0)
                )
                quality_loss = F.binary_cross_entropy_with_logits(
                    quality_logits, keep_targets
                )
                loss = positive_loss + 0.08 * negative_loss + 0.30 * quality_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
        parallel_model.eval()
        dice_sum = positive_count = correct = sample_count = 0
        with torch.no_grad():
            for features, targets, keep_targets in validation_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                keep_targets = keep_targets.to(device, non_blocking=True)
                mask_logits, quality_logits = parallel_model(features)
                predictions = mask_logits.sigmoid() >= 0.50
                positive = keep_targets.bool()
                if positive.any():
                    target_masks = targets[positive] >= 0.50
                    pred_masks = predictions[positive]
                    intersection = (pred_masks & target_masks).sum(dim=(1, 2, 3)).float()
                    denominator = pred_masks.sum(dim=(1, 2, 3)) + target_masks.sum(
                        dim=(1, 2, 3)
                    )
                    dice_sum += float(
                        ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum()
                    )
                    positive_count += int(positive.sum())
                correct += int(
                    ((quality_logits.sigmoid() >= 0.50) == positive).sum()
                )
                sample_count += len(features)
        validation_dice = dice_sum / max(positive_count, 1)
        quality_accuracy = correct / max(sample_count, 1)
        validation_score = validation_dice + 0.10 * quality_accuracy
        row = {
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "validation_dice": validation_dice,
            "quality_accuracy": quality_accuracy,
            "selection_score": validation_score,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_minutes": (time.monotonic() - started) / 60,
        }
        history.append(row)
        print(json.dumps({"event": "proposal_epoch", **row}, separators=(",", ":")), flush=True)
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(fold_dir / "history.csv", index=False)
    del parallel_model, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint, history_frame


@torch.no_grad()
def proposal_v2_probability_records(
    image: np.ndarray,
    seeds: Sequence[np.ndarray],
    detector_scores: Sequence[float],
    model: ProposalRefinerV2,
    device: torch.device,
    cfg: CFG,
) -> list[dict[str, Any]]:
    if not seeds:
        return []
    size = (cfg.refiner_crop,) * 2
    bounds: list[tuple[int, int, int, int]] = []
    feature_arrays = []
    for seed in seeds:
        x0, y0, x1, y1 = square_bounds(
            seed, cfg.refiner_context, cfg.refiner_min_crop
        )
        bounds.append((x0, y0, x1, y1))
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1].astype(np.uint8), size,
            interpolation=cv2.INTER_NEAREST,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        feature_arrays.append(
            np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32)
            / 255.0
        )
    tensor = torch.from_numpy(np.stack(feature_arrays)).to(device)
    transforms = [
        lambda value: value,
        lambda value: value.flip(-1),
        lambda value: value.flip(-2),
        lambda value: value.flip((-2, -1)),
    ]
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    mask_chunks, quality_chunks = [], []
    for chunk in transformed.split(12):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mask_logits, quality_logits = model(chunk)
        mask_chunks.append(mask_logits.sigmoid().float().cpu())
        quality_chunks.append(quality_logits.sigmoid().float().cpu())
    mask_probabilities = torch.cat(mask_chunks, dim=0)
    quality_probabilities = torch.cat(quality_chunks, dim=0)
    count = len(seeds)
    records = []
    for index, (seed, detector_score, crop_bounds) in enumerate(
        zip(seeds, detector_scores, bounds)
    ):
        views = [
            transforms[view](
                mask_probabilities[
                    view * count + index : view * count + index + 1
                ]
            )[0, 0].numpy()
            for view in range(4)
        ]
        quality = float(
            np.mean(
                [quality_probabilities[view * count + index].item() for view in range(4)]
            )
        )
        x0, y0, x1, y1 = crop_bounds
        records.append(
            {
                "seed": seed,
                "detector_score": float(detector_score),
                "quality": quality,
                "bounds": crop_bounds,
                "probability": cv2.resize(
                    np.mean(views, axis=0),
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR,
                ),
            }
        )
    return records


def predictions_for_proposal_setting(
    records: Sequence[Mapping[str, Any]],
    setting: ProposalV2Setting,
    minimum_area: int,
) -> tuple[list[np.ndarray], list[float]]:
    candidates: list[tuple[np.ndarray, float]] = []
    for record in records:
        if float(record["quality"]) < setting.quality_threshold:
            continue
        crop_mask = record["probability"] >= setting.mask_threshold
        if int(crop_mask.sum()) < minimum_area:
            continue
        full_mask = np.zeros_like(record["seed"], dtype=bool)
        x0, y0, x1, y1 = record["bounds"]
        full_mask[y0:y1, x0:x1] = crop_mask
        foreground_quality = float(record["probability"][crop_mask].mean())
        score = (
            float(record["detector_score"])
            * float(record["quality"])
            * foreground_quality
        )
        candidates.append((full_mask, score))
    if not candidates and records:
        # Preserve the competition's one-image fallback without weakening the
        # quality filter for ordinary images.
        record = max(records, key=lambda value: float(value["quality"]))
        crop_mask = record["probability"] >= setting.mask_threshold
        if crop_mask.any():
            full_mask = np.zeros_like(record["seed"], dtype=bool)
            x0, y0, x1, y1 = record["bounds"]
            full_mask[y0:y1, x0:x1] = crop_mask
            candidates = [(full_mask, 0.001 * float(record["quality"]))]
    if not candidates:
        return [], []
    masks, scores = zip(*candidates)
    return resolve_overlaps(list(masks), list(scores), minimum_area)


def proposal_settings() -> tuple[ProposalV2Setting, ...]:
    return tuple(
        ProposalV2Setting(quality, mask)
        for quality in (0.20, 0.35, 0.50, 0.65)
        for mask in (0.45, 0.50, 0.55, 0.60, 0.65)
    )


def catalog_by_filename(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        output.setdefault(str(sample["filename"]), []).append(sample)
    return output


def official_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals = {
        key: float(sum(float(row[key]) for row in rows))
        for key in ("sum_iou", "tp", "fp", "fn")
    }
    denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
    return {
        **totals,
        "official_pq": totals["sum_iou"] / denominator if denominator else 0.0,
        "rq": totals["tp"] / denominator if denominator else 0.0,
        "mean_predictions": float(np.mean([float(row["n_pred"]) for row in rows])),
    }


def evaluate_proposal_refiner_oof(
    fold: int,
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    indices: Mapping[str, Any],
    checkpoint: Path,
    baseline_checkpoint: Path,
    cfg: CFG,
    settings: Sequence[ProposalV2Setting],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    model = ProposalRefinerV2()
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device).eval()
    baseline = LegacySeedRefiner()
    baseline.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    baseline.to(device).eval()
    baseline_cfg = legacy_inference_cfg()
    grouped = catalog_by_filename(validation_samples)
    setting_rows: dict[str, list[dict[str, Any]]] = {
        setting.name: [] for setting in settings
    }
    baseline_rows: list[dict[str, Any]] = []
    baseline_setting = AuditSetting(
        "seed_nms_0.45",
        detector_conf=0.0,
        seed_nms_iou=cfg.seed_nms_iou,
        refiner_threshold=0.65,
        minimum_area=cfg.minimum_area,
    )
    for number, (filename, samples) in enumerate(sorted(grouped.items()), 1):
        image = cv2.imread(str(image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / filename)
        seeds = [mask_utils.decode(sample["seed_rle"]).astype(bool) for sample in samples]
        detector_scores = [float(sample["detector_score"]) for sample in samples]
        v2_records = proposal_v2_probability_records(
            image, seeds, detector_scores, model, device, cfg
        )
        legacy_records = legacy_probability_records(
            image, seeds, detector_scores, baseline, device, baseline_cfg
        )
        baseline_masks, _ = predictions_for_setting(legacy_records, baseline_setting)
        ground_truth = ground_truth_sets(filename, indices)
        for annotator, gt_masks in enumerate(ground_truth):
            baseline_rows.append(
                audit_metric_row(
                    filename, fold, annotator, "baseline", gt_masks, baseline_masks
                )
            )
        for setting in settings:
            masks, _ = predictions_for_proposal_setting(
                v2_records, setting, cfg.minimum_area
            )
            for annotator, gt_masks in enumerate(ground_truth):
                setting_rows[setting.name].append(
                    audit_metric_row(
                        filename, fold, annotator, setting.name, gt_masks, masks
                    )
                )
        if number % 20 == 0 or number == len(grouped):
            print(
                {
                    "stage": "proposal_v2_oof",
                    "fold": fold,
                    "image": f"{number}/{len(grouped)}",
                },
                flush=True,
            )
    summaries = {
        name: official_summary(rows) for name, rows in setting_rows.items()
    }
    best_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["official_pq"],
            -summaries[name]["mean_predictions"],
        ),
    )
    fold_dir = output_dir / "oof" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(fold_dir / "baseline_by_annotator.csv", index=False)
    pd.DataFrame(setting_rows[best_name]).to_csv(
        fold_dir / "proposal_v2_best_by_annotator.csv", index=False
    )
    report = {
        "fold": fold,
        "baseline": official_summary(baseline_rows),
        "settings": summaries,
        "best_setting": best_name,
        "best": summaries[best_name],
        "official_pq_delta": (
            summaries[best_name]["official_pq"]
            - official_summary(baseline_rows)["official_pq"]
        ),
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    del model, baseline
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"event": "proposal_v2_fold_complete", **report}, separators=(",", ":")), flush=True)
    return report


def parse_proposal_setting(name: str) -> ProposalV2Setting:
    quality_text, mask_text = name.split("_")
    return ProposalV2Setting(
        float(quality_text.removeprefix("q")),
        float(mask_text.removeprefix("m")),
    )


@torch.no_grad()
def predict_proposal_v2_instances(
    image_path: Path,
    detector: Any,
    model: ProposalRefinerV2,
    device: torch.device,
    cfg: CFG,
    setting: ProposalV2Setting,
) -> tuple[list[np.ndarray], list[float]]:
    image, seeds, scores = raw_legacy_candidates(image_path, detector, device, cfg)
    if not seeds:
        seeds = [classical_fallback_seed(image, cfg.minimum_area)]
        scores = [0.01]
    selected = selected_proposal_indices(seeds, scores, cfg)
    chosen_seeds = [seeds[index] for index in selected]
    chosen_scores = [float(scores[index]) for index in selected]
    records = proposal_v2_probability_records(
        image, chosen_seeds, chosen_scores, model, device, cfg
    )
    return predictions_for_proposal_setting(records, setting, cfg.minimum_area)


def run_proposal_v2_test(
    paths: Mapping[str, Path],
    baseline_root: Path,
    output_dir: Path,
    cfg: CFG,
    setting: ProposalV2Setting,
) -> dict[str, Any]:
    from ultralytics import YOLO

    models = []
    for fold in range(cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(str(baseline_root / "models" / f"fold_{fold}" / "detector.pt"))
        model = ProposalRefinerV2()
        model.load_state_dict(
            torch.load(
                output_dir / "models" / f"fold_{fold}" / "proposal_refiner_v2.pt",
                map_location=device,
            ),
            strict=True,
        )
        models.append((fold, detector, model.to(device).eval(), device))
    policy = FusionPolicy(
        "s3_medoid",
        match_iou=0.20,
        minimum_support=3,
        mask_mode="medoid",
        minimum_area=cfg.minimum_area,
    )
    rows: list[dict[str, str]] = []
    diagnostics = []
    test_files = sorted(paths["test"].glob("*.jpeg"))
    for number, image_path in enumerate(test_files, 1):
        predictions = []
        for fold, detector, model, device in models:
            masks, scores = predict_proposal_v2_instances(
                image_path, detector, model, device, cfg, setting
            )
            predictions.extend(
                (fold, mask, score) for mask, score in zip(masks, scores)
            )
        masks, _, detail = fuse_predictions_with_policy(predictions, policy)
        for instance_index, mask in enumerate(masks, 1):
            rows.append(
                {
                    "filament_id": f"{image_path.stem}_{instance_index}",
                    "segmentation_rle": encode_mask(mask),
                }
            )
        diagnostics.append(
            {
                "filename": image_path.name,
                "raw_instances": len(predictions),
                "instances": len(masks),
                "mean_area": float(np.mean([mask.sum() for mask in masks])),
                "fallback": bool(detail["fallback"]),
                "support_histogram": json.dumps(
                    detail["support_histogram"], separators=(",", ":"), sort_keys=True
                ),
            }
        )
        if number % 10 == 0 or number == len(test_files):
            print(
                {
                    "stage": "proposal_v2_test",
                    "image": f"{number}/{len(test_files)}",
                    "rows": len(rows),
                },
                flush=True,
            )
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    test_ids = {path.stem for path in test_files}
    submitted_ids = set(submission["filament_id"].str.rsplit("_", n=1).str[0])
    if submitted_ids != test_ids:
        raise RuntimeError(f"Proposal v2 misses {len(test_ids - submitted_ids)} test images")
    if submission.empty or not submission["filament_id"].is_unique:
        raise RuntimeError("Invalid proposal v2 submission")
    submission_path = output_dir / "submission_proposal_v2_s3_medoid.csv"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "test_diagnostics.csv", index=False)
    return {
        "path": str(submission_path),
        "rows": len(submission),
        "images": len(submitted_ids),
        "mean_instances": len(submission) / len(submitted_ids),
        "fallback_images": int(sum(row["fallback"] for row in diagnostics)),
    }


def run_proposal_refiner_v2(
    paths: Mapping[str, Path],
    baseline_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Leakage-safe auto-gated proposal refiner experiment and production run."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(baseline_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.refiner_crop = 384
    cfg.refiner_min_crop = 128
    cfg.refiner_batch = 8
    cfg.refiner_epochs = 8
    cfg.num_workers = 4
    cfg.seed_nms_iou = 0.45
    all_settings = proposal_settings()
    selected_setting: ProposalV2Setting | None = None
    fold_reports: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    for fold in range(cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(str(baseline_root / "models" / f"fold_{fold}" / "detector.pt"))
        filenames = manifest["filename"].tolist()
        samples = build_proposal_catalog(
            filenames,
            paths["train"],
            detector,
            device,
            medoids,
            indices,
            cfg,
            fold,
        )
        train_files = set(
            manifest.loc[manifest["fold"] != fold, "filename"].astype(str)
        )
        validation_files = set(
            manifest.loc[manifest["fold"] == fold, "filename"].astype(str)
        )
        train_samples = [sample for sample in samples if sample["filename"] in train_files]
        validation_samples = [
            sample for sample in samples if sample["filename"] in validation_files
        ]
        del detector
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint, history = train_proposal_refiner(
            fold,
            train_samples,
            validation_samples,
            paths["train"],
            baseline_root / "models" / f"fold_{fold}" / "refiner.pt",
            cfg,
            output_dir,
        )
        settings = all_settings if selected_setting is None else (selected_setting,)
        report = evaluate_proposal_refiner_oof(
            fold,
            validation_samples,
            paths["train"],
            indices,
            checkpoint,
            baseline_root / "models" / f"fold_{fold}" / "refiner.pt",
            cfg,
            settings,
            output_dir,
        )
        fold_reports[str(fold)] = {
            **report,
            "best_crop_dice": float(history["validation_dice"].max()),
            "runtime_hours": (time.monotonic() - run_started) / 3600,
        }
        if selected_setting is None:
            selected_setting = parse_proposal_setting(report["best_setting"])
            # A full run is promoted only for a material, end-to-end held-out gain.
            if report["official_pq_delta"] < 0.008 or (
                report["best"]["rq"] + 0.005 < report["baseline"]["rq"]
            ):
                result = {
                    "status": "rejected_after_fold0",
                    "gate_delta": 0.008,
                    "folds": fold_reports,
                    "selected_setting": asdict(selected_setting),
                }
                (output_dir / "proposal_v2_report.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                print(json.dumps({"event": "proposal_v2_rejected", **result}, separators=(",", ":")), flush=True)
                return result
        fold_best_path = output_dir / "oof" / f"fold_{fold}" / "proposal_v2_best_by_annotator.csv"
        oof_rows.extend(pd.read_csv(fold_best_path).to_dict("records"))
        elapsed_hours = (time.monotonic() - run_started) / 3600
        if elapsed_hours > 5.25 and fold < cfg.n_folds - 1:
            result = {
                "status": "checkpoints_partial_runtime_guard",
                "runtime_hours": elapsed_hours,
                "folds": fold_reports,
                "selected_setting": asdict(selected_setting),
            }
            (output_dir / "proposal_v2_report.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps({"event": "proposal_v2_runtime_guard", **result}, separators=(",", ":")), flush=True)
            return result
    aggregate = official_summary(oof_rows)
    if aggregate["official_pq"] < 0.425:
        result = {
            "status": "rejected_after_full_oof",
            "promotion_pq": 0.425,
            "aggregate_oof": aggregate,
            "folds": fold_reports,
            "selected_setting": asdict(selected_setting),
        }
        (output_dir / "proposal_v2_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({"event": "proposal_v2_full_rejected", **result}, separators=(",", ":")), flush=True)
        return result
    elapsed_hours = (time.monotonic() - run_started) / 3600
    if elapsed_hours > 5.25:
        result = {
            "status": "checkpoints_complete_runtime_guard",
            "runtime_hours": elapsed_hours,
            "aggregate_oof": aggregate,
            "folds": fold_reports,
            "selected_setting": asdict(selected_setting),
        }
        (output_dir / "proposal_v2_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({"event": "proposal_v2_runtime_guard", **result}, separators=(",", ":")), flush=True)
        return result
    test = run_proposal_v2_test(
        paths, baseline_root, output_dir, cfg, selected_setting
    )
    result = {
        "status": "submission_ready",
        "aggregate_oof": aggregate,
        "folds": fold_reports,
        "selected_setting": asdict(selected_setting),
        "test": test,
        "runtime_hours": (time.monotonic() - run_started) / 3600,
    }
    (output_dir / "proposal_v2_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "proposal_v2_complete", **result}, separators=(",", ":")), flush=True)
    return result


V3_METADATA_DIM = 6


def discover_proposal_v2_root() -> Path | None:
    """Locate the promoted v2 checkpoints attached as a kernel source."""
    candidates = sorted(
        {
            path.parents[2]
            for path in Path("/kaggle/input").glob(
                "**/proposal_refiner_v2/models/fold_0/proposal_refiner_v2.pt"
            )
        }
    )
    valid = [
        root
        for root in candidates
        if all(
            (root / "models" / f"fold_{fold}" / "proposal_refiner_v2.pt").is_file()
            for fold in range(5)
        )
    ]
    if len(valid) > 1:
        raise RuntimeError(f"Multiple proposal v2 roots found: {valid}")
    return valid[0] if valid else None


@torch.no_grad()
def run_proposal_v2_policy_sweep(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v2_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Inference-only sweep of four five-fold fusion policies for promoted V2."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = legacy_inference_cfg()
    cfg.refiner_crop = 384
    cfg.refiner_min_crop = 128
    cfg.seed_nms_iou = 0.45
    cfg.minimum_area = 96
    setting = ProposalV2Setting(quality_threshold=0.50, mask_threshold=0.50)
    policies = (
        FusionPolicy(
            "s2_medoid",
            match_iou=0.20,
            minimum_support=2,
            mask_mode="medoid",
            minimum_area=cfg.minimum_area,
        ),
        FusionPolicy(
            "s3_majority",
            match_iou=0.20,
            minimum_support=3,
            mask_mode="majority",
            minimum_area=cfg.minimum_area,
        ),
        FusionPolicy(
            "s4_medoid",
            match_iou=0.20,
            minimum_support=4,
            mask_mode="medoid",
            minimum_area=cfg.minimum_area,
        ),
        FusionPolicy(
            "s5_medoid",
            match_iou=0.20,
            minimum_support=5,
            mask_mode="medoid",
            minimum_area=cfg.minimum_area,
        ),
    )
    models = []
    for fold in range(cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(
            str(baseline_root / "models" / f"fold_{fold}" / "detector.pt")
        )
        model = ProposalRefinerV2()
        checkpoint = (
            v2_root
            / "models"
            / f"fold_{fold}"
            / "proposal_refiner_v2.pt"
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location=device), strict=True
        )
        models.append((fold, detector, model.to(device).eval(), device))
    rows_by_policy: dict[str, list[dict[str, str]]] = {
        policy.name: [] for policy in policies
    }
    diagnostics_by_policy: dict[str, list[dict[str, Any]]] = {
        policy.name: [] for policy in policies
    }
    test_files = sorted(paths["test"].glob("*.jpeg"))
    if len(test_files) != 180:
        raise RuntimeError(f"Expected 180 test images, found {len(test_files)}")
    for number, image_path in enumerate(test_files, 1):
        predictions = []
        for fold, detector, model, device in models:
            masks, scores = predict_proposal_v2_instances(
                image_path, detector, model, device, cfg, setting
            )
            predictions.extend(
                (fold, mask, score) for mask, score in zip(masks, scores)
            )
        for policy in policies:
            masks, _, detail = fuse_predictions_with_policy(predictions, policy)
            for instance_index, mask in enumerate(masks, 1):
                rows_by_policy[policy.name].append(
                    {
                        "filament_id": f"{image_path.stem}_{instance_index}",
                        "segmentation_rle": encode_mask(mask),
                    }
                )
            diagnostics_by_policy[policy.name].append(
                {
                    "filename": image_path.name,
                    "raw_instances": len(predictions),
                    "instances": len(masks),
                    "mean_area": float(
                        np.mean([mask.sum() for mask in masks])
                    ),
                    "fallback": bool(detail["fallback"]),
                    "support_histogram": json.dumps(
                        detail["support_histogram"],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            )
        if number % 10 == 0 or number == len(test_files):
            print(
                {
                    "stage": "proposal_v2_policy_sweep",
                    "image": f"{number}/{len(test_files)}",
                    "rows": {
                        name: len(rows) for name, rows in rows_by_policy.items()
                    },
                },
                flush=True,
            )
    test_ids = {path.stem for path in test_files}
    variants: dict[str, Any] = {}
    for policy in policies:
        name = policy.name
        submission = pd.DataFrame(
            rows_by_policy[name],
            columns=["filament_id", "segmentation_rle"],
        )
        submitted_ids = set(
            submission["filament_id"].str.rsplit("_", n=1).str[0]
        )
        if submitted_ids != test_ids:
            raise RuntimeError(
                f"Policy {name} misses {len(test_ids - submitted_ids)} test images"
            )
        if submission.empty or not submission["filament_id"].is_unique:
            raise RuntimeError(f"Invalid submission for policy {name}")
        submission_path = output_dir / f"submission_v2_{name}.csv"
        diagnostics_path = output_dir / f"diagnostics_v2_{name}.csv"
        submission.to_csv(submission_path, index=False)
        pd.DataFrame(diagnostics_by_policy[name]).to_csv(
            diagnostics_path, index=False
        )
        variants[name] = {
            "path": str(submission_path),
            "rows": len(submission),
            "images": len(submitted_ids),
            "mean_instances": len(submission) / len(submitted_ids),
            "fallback_images": int(
                sum(row["fallback"] for row in diagnostics_by_policy[name])
            ),
        }
    result = {
        "status": "submission_sweep_ready",
        "model": "proposal_refiner_v2_five_fold",
        "strict_oof_reference_pq": 0.43382642410982014,
        "known_s3_medoid_public_lb": 0.39,
        "setting": asdict(setting),
        "variants": variants,
        "runtime_hours": (time.monotonic() - run_started) / 3600,
        "submitted": False,
    }
    (output_dir / "policy_sweep_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "proposal_v2_policy_sweep_complete", **result},
            separators=(",", ":"),
        ),
        flush=True,
    )
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return result


def v3_consensus_clusters_for_file(
    filename: str,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    minimum_iou: float,
) -> list[dict[str, Any]]:
    """Align every annotation version into instance-level uncertainty targets."""
    image_ids = list(indices["image_ids_by_file"][filename])
    medoid_id = medoids[filename]
    ordered_ids = [medoid_id] + [value for value in image_ids if value != medoid_id]
    info = indices["images_by_id"][medoid_id]
    height, width = int(info["height"]), int(info["width"])
    clusters: list[dict[str, Any]] = []
    for annotator, image_id in enumerate(ordered_ids):
        masks = [
            annotation_to_mask(annotation, height, width)
            for annotation in indices["annotations_by_image"].get(image_id, [])
            if not annotation.get("iscrowd", 0)
        ]
        if not clusters:
            clusters = [
                {"members": [(annotator, mask)], "representative": mask}
                for mask in masks
            ]
            continue
        representatives = [cluster["representative"] for cluster in clusters]
        overlaps = pairwise_iou(representatives, masks)
        matched: set[int] = set()
        if overlaps.size:
            cluster_indices, mask_indices = linear_sum_assignment(-overlaps)
            for cluster_index, mask_index in zip(cluster_indices, mask_indices):
                if overlaps[int(cluster_index), int(mask_index)] >= minimum_iou:
                    clusters[int(cluster_index)]["members"].append(
                        (annotator, masks[int(mask_index)])
                    )
                    matched.add(int(mask_index))
        for mask_index, mask in enumerate(masks):
            if mask_index not in matched:
                clusters.append(
                    {"members": [(annotator, mask)], "representative": mask}
                )

    output: list[dict[str, Any]] = []
    for cluster in clusters:
        member_masks = [member[1] for member in cluster["members"]]
        if len(member_masks) > 1:
            overlaps = pairwise_iou(member_masks, member_masks)
            representative_index = int(np.argmax(overlaps.mean(axis=1)))
        else:
            representative_index = 0
        output.append(
            {
                "representative": member_masks[representative_index],
                "target_rles": [binary_rle(mask) for mask in member_masks],
                "existence": len({member[0] for member in cluster["members"]})
                / max(len(image_ids), 1),
                "n_annotators": len(image_ids),
            }
        )
    return output


def selected_v3_proposal_indices(
    seeds: Sequence[np.ndarray], scores: Sequence[float], cfg: CFG
) -> list[int]:
    records = [
        {"seed": seed, "detector_score": float(score)}
        for seed, score in zip(seeds, scores)
    ]
    setting = AuditSetting(
        "proposal_v3_high_recall",
        detector_conf=0.05,
        seed_nms_iou=cfg.seed_nms_iou,
        refiner_threshold=0.50,
        minimum_area=cfg.minimum_area,
    )
    return audit_candidate_indices(records, setting)[:24]


@torch.no_grad()
def build_v3_samples_for_file(
    filename: str,
    image_dir: Path,
    detector: Any,
    device: torch.device,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
    fold: int,
) -> list[dict[str, Any]]:
    image, seeds, scores = raw_legacy_candidates(
        image_dir / filename, detector, device, cfg
    )
    if not seeds:
        seeds = [classical_fallback_seed(image, cfg.minimum_area)]
        scores = [0.01]
    selected = selected_v3_proposal_indices(seeds, scores, cfg)
    chosen_seeds = [seeds[index] for index in selected]
    chosen_scores = [float(scores[index]) for index in selected]
    clusters = v3_consensus_clusters_for_file(
        filename, medoids, indices, cfg.consensus_match_iou
    )
    representatives = [cluster["representative"] for cluster in clusters]
    overlaps = pairwise_iou(representatives, chosen_seeds)
    assigned: dict[int, tuple[int, float]] = {}
    if overlaps.size:
        cluster_indices, seed_indices = linear_sum_assignment(-overlaps)
        for cluster_index, seed_index in zip(cluster_indices, seed_indices):
            overlap = float(overlaps[int(cluster_index), int(seed_index)])
            if overlap >= 0.01:
                assigned[int(seed_index)] = (int(cluster_index), overlap)
    samples: list[dict[str, Any]] = []
    for seed_index, (seed, detector_score) in enumerate(
        zip(chosen_seeds, chosen_scores)
    ):
        match = assigned.get(seed_index)
        cluster = clusters[match[0]] if match is not None else None
        samples.append(
            {
                "filename": filename,
                "fold": fold,
                "seed_rle": binary_rle(seed),
                "target_rles": cluster["target_rles"] if cluster is not None else [],
                "existence": float(cluster["existence"]) if cluster is not None else 0.0,
                "n_annotators": int(cluster["n_annotators"]) if cluster is not None else len(
                    indices["image_ids_by_file"][filename]
                ),
                "detector_score": detector_score,
                "seed_target_iou": match[1] if match is not None else 0.0,
            }
        )
    return samples


def build_v3_oof_catalog(
    manifest: pd.DataFrame,
    image_dir: Path,
    baseline_root: Path,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
) -> list[dict[str, Any]]:
    """Generate every training proposal with the detector that excluded it."""
    from ultralytics import YOLO

    samples: list[dict[str, Any]] = []
    for fold in range(cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(str(baseline_root / "models" / f"fold_{fold}" / "detector.pt"))
        filenames = manifest.loc[
            manifest["fold"] == fold, "filename"
        ].astype(str).tolist()
        fold_start = len(samples)
        for number, filename in enumerate(filenames, 1):
            samples.extend(
                build_v3_samples_for_file(
                    filename,
                    image_dir,
                    detector,
                    device,
                    medoids,
                    indices,
                    cfg,
                    fold,
                )
            )
            if number % 25 == 0 or number == len(filenames):
                fold_samples = samples[fold_start:]
                print(
                    {
                        "stage": "proposal_v3_oof_catalog",
                        "fold": fold,
                        "image": f"{number}/{len(filenames)}",
                        "samples": len(fold_samples),
                        "matched": sum(float(x["existence"]) > 0 for x in fold_samples),
                    },
                    flush=True,
                )
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    return samples


def balanced_v3_training_samples(
    samples: Sequence[Mapping[str, Any]], maximum_negatives_per_image: int = 4
) -> list[Mapping[str, Any]]:
    grouped = catalog_by_filename(samples)
    output: list[Mapping[str, Any]] = []
    for filename in sorted(grouped):
        positives = [value for value in grouped[filename] if float(value["existence"]) > 0]
        negatives = sorted(
            (value for value in grouped[filename] if float(value["existence"]) == 0),
            key=lambda value: float(value["detector_score"]),
            reverse=True,
        )
        output.extend(positives)
        output.extend(negatives[:maximum_negatives_per_image])
    return output


def proposal_v3_metadata(
    image: np.ndarray, seed: np.ndarray, detector_score: float
) -> np.ndarray:
    ys, xs = np.where(seed)
    pixels = max(seed.size, 1)
    if len(xs):
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        box_area = max(width * height, 1)
        center_x = float((xs.min() + xs.max()) / 2 / max(seed.shape[1] - 1, 1))
        center_y = float((ys.min() + ys.max()) / 2 / max(seed.shape[0] - 1, 1))
        radius = min(math.hypot(2 * center_x - 1, 2 * center_y - 1), 1.5) / 1.5
        fill = float(seed.sum() / box_area)
        aspect = float(math.tanh(abs(math.log(max(width, 1) / max(height, 1)))))
        contrast = float(np.clip((float(image.mean()) - float(image[seed].mean())) / 128, -1, 1))
    else:
        radius = fill = aspect = contrast = 0.0
    area_fraction = max(float(seed.sum() / pixels), 1e-8)
    log_area = float(np.clip((math.log10(area_fraction) + 4.0) / 4.0, -1, 1))
    return np.asarray(
        [float(detector_score), log_area, fill, aspect, radius, contrast],
        dtype=np.float32,
    )


class ProposalRefinerV3Dataset(Dataset):
    """OOF proposals with soft multi-annotator shape and existence targets."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        image_dir: Path,
        cfg: CFG,
        augment: bool,
    ) -> None:
        self.samples = list(samples)
        self.image_dir = image_dir
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(
            str(self.image_dir / str(sample["filename"])), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise FileNotFoundError(self.image_dir / str(sample["filename"]))
        seed = mask_utils.decode(sample["seed_rle"]).astype(bool)
        target_rles = list(sample["target_rles"])
        if target_rles:
            target = np.mean(
                [mask_utils.decode(value).astype(np.float32) for value in target_rles],
                axis=0,
            ).astype(np.float32)
        else:
            target = np.zeros_like(seed, dtype=np.float32)
        x0, y0, x1, y1 = square_bounds(
            seed, self.cfg.refiner_context, self.cfg.refiner_min_crop
        )
        size = (self.cfg.refiner_crop,) * 2
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1].astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        )
        target_crop = cv2.resize(
            target[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        features = np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(
            np.float32
        ) / 255.0
        metadata = proposal_v3_metadata(
            image, seed, float(sample["detector_score"])
        )
        if self.augment:
            rotation = random.randrange(4)
            features = np.rot90(features, rotation, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, rotation).copy()
            if random.random() < 0.5:
                features = features[:, :, ::-1].copy()
                target_crop = target_crop[:, ::-1].copy()
            if random.random() < 0.5:
                features = features[:, ::-1, :].copy()
                target_crop = target_crop[::-1, :].copy()
            gain = random.uniform(0.82, 1.18)
            bias = random.uniform(-0.05, 0.05)
            features[:2] = np.clip(features[:2] * gain + bias, 0, 1)
        return (
            torch.from_numpy(features),
            torch.from_numpy(target_crop[None].astype(np.float32)),
            torch.tensor(float(sample["existence"]), dtype=torch.float32),
            torch.from_numpy(metadata),
        )


class ProposalRefinerV3(ProposalRefinerV2):
    """V2 masks plus calibrated existence and predicted-mask IoU heads."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata_project = nn.Sequential(
            nn.LayerNorm(V3_METADATA_DIM),
            nn.Linear(V3_METADATA_DIM, 32),
            nn.SiLU(inplace=True),
        )
        self.existence_correction = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(512 + 32 + 32 + 2, 160),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(160, 1),
        )
        nn.init.zeros_(self.existence_correction[-1].weight)
        nn.init.zeros_(self.existence_correction[-1].bias)

    def forward(
        self, tensor: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = (tensor - self.mean) / self.std
        x0 = self.stem(tensor)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        legacy_existence = self.quality_head(x4).squeeze(1)
        metadata_features = self.metadata_project(metadata)
        existence = legacy_existence + self.existence_correction(
            metadata_features
        ).squeeze(1)
        x = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x1], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x0], dim=1))
        mask_logits = self.head(self.up0(x))
        probabilities = mask_logits.sigmoid()
        mask_statistics = torch.cat(
            [
                probabilities.mean(dim=(2, 3)),
                (probabilities * (1.0 - probabilities)).mean(dim=(2, 3)),
            ],
            dim=1,
        )
        iou_features = torch.cat(
            [
                F.adaptive_avg_pool2d(x4, 1).flatten(1),
                F.adaptive_avg_pool2d(x, 1).flatten(1),
                metadata_features,
                mask_statistics,
            ],
            dim=1,
        )
        iou = self.iou_head(iou_features).squeeze(1)
        return mask_logits, existence, iou


def initialize_proposal_refiner_v3(
    checkpoint: Path, device: torch.device
) -> ProposalRefinerV3:
    model = ProposalRefinerV3()
    result = model.load_state_dict(
        torch.load(checkpoint, map_location="cpu"), strict=False
    )
    allowed_prefixes = (
        "metadata_project.", "existence_correction.", "iou_head."
    )
    unexpected_missing = [
        key for key in result.missing_keys if not key.startswith(allowed_prefixes)
    ]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected v3 initialization: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model.to(device)


def soft_skeletonize(probability: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    def erode(value: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-value, 3, stride=1, padding=1)

    def dilate(value: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(value, 3, stride=1, padding=1)

    def opened(value: torch.Tensor) -> torch.Tensor:
        return dilate(erode(value))

    skeleton = F.relu(probability - opened(probability))
    value = probability
    for _ in range(iterations):
        value = erode(value)
        delta = F.relu(value - opened(value))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_losses(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    probabilities = F.interpolate(
        probabilities, size=(224, 224), mode="bilinear", align_corners=False
    )
    targets = F.interpolate(
        targets, size=(224, 224), mode="bilinear", align_corners=False
    )
    predicted_skeleton = soft_skeletonize(probabilities)
    target_skeleton = soft_skeletonize(targets)
    dimensions = (1, 2, 3)
    topology_precision = (
        (predicted_skeleton * targets).sum(dimensions) + 1.0
    ) / (predicted_skeleton.sum(dimensions) + 1.0)
    topology_sensitivity = (
        (target_skeleton * probabilities).sum(dimensions) + 1.0
    ) / (target_skeleton.sum(dimensions) + 1.0)
    return 1.0 - (
        2.0 * topology_precision * topology_sensitivity + 1e-6
    ) / (topology_precision + topology_sensitivity + 1e-6)


def dynamic_mask_iou_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        predictions = logits.sigmoid() >= 0.50
        truth = targets >= 0.50
        intersection = (predictions & truth).sum(dim=(1, 2, 3)).float()
        union = (predictions | truth).sum(dim=(1, 2, 3)).float()
        return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(union))


def train_proposal_refiner_v3(
    fold: int,
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    v2_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    fold_dir = output_dir / "models" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "proposal_refiner_v3.pt"
    train_dataset = ProposalRefinerV3Dataset(train_samples, image_dir, cfg, True)
    validation_dataset = ProposalRefinerV3Dataset(
        validation_samples, image_dir, cfg, False
    )
    gpu_count = torch.cuda.device_count()
    effective_batch = cfg.refiner_batch * max(gpu_count, 1)
    loader_options = {
        "num_workers": min(cfg.num_workers, 4),
        "pin_memory": True,
        "persistent_workers": min(cfg.num_workers, 4) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 100 + fold),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=effective_batch * 2,
        shuffle=False,
        **loader_options,
    )
    device = torch.device("cuda:0")
    model = initialize_proposal_refiner_v3(v2_checkpoint, device)
    parallel_model: nn.Module = (
        nn.DataParallel(model, device_ids=list(range(gpu_count)))
        if gpu_count > 1 else model
    )
    new_modules = (
        model.metadata_project, model.existence_correction, model.iou_head
    )
    new_parameter_ids = {
        id(parameter)
        for module in new_modules
        for parameter in module.parameters()
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if id(parameter) not in new_parameter_ids
                ],
                "lr": 5e-5,
            },
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if id(parameter) in new_parameter_ids
                ],
                "lr": 2e-4,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.refiner_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_validation_score = -1.0
    print(
        {
            "stage": "proposal_v3_train",
            "fold": fold,
            "gpus": gpu_count,
            "effective_batch": effective_batch,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "train_matched": sum(float(x["existence"]) > 0 for x in train_samples),
        },
        flush=True,
    )
    for epoch in range(cfg.refiner_epochs):
        epoch_started = time.monotonic()
        parallel_model.train()
        loss_sum = 0.0
        for features, targets, existence_targets, metadata in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            existence_targets = existence_targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                mask_logits, existence_logits, iou_logits = parallel_model(
                    features, metadata
                )
                boundary_bce, dice = proposal_mask_losses(mask_logits, targets)
                topology = soft_cldice_losses(mask_logits.sigmoid(), targets)
                weights = existence_targets
                mask_loss = (
                    ((0.40 * boundary_bce + 0.50 * dice + 0.10 * topology) * weights).sum()
                    / weights.sum().clamp_min(1.0)
                )
                negatives = (existence_targets == 0).float()
                negative_mask_loss = (
                    (boundary_bce * negatives).sum() / negatives.sum().clamp_min(1.0)
                )
                existence_loss = F.binary_cross_entropy_with_logits(
                    existence_logits, existence_targets
                )
                iou_targets = dynamic_mask_iou_targets(mask_logits, targets)
                positives = (existence_targets > 0).float()
                iou_loss = (
                    (
                        F.smooth_l1_loss(
                            iou_logits.sigmoid(), iou_targets, reduction="none"
                        ) * positives
                    ).sum()
                    / positives.sum().clamp_min(1.0)
                )
                loss = (
                    mask_loss
                    + 0.04 * negative_mask_loss
                    + 0.35 * existence_loss
                    + 0.25 * iou_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())

        parallel_model.eval()
        dice_sum = existence_error = iou_error = 0.0
        positive_count = sample_count = 0
        with torch.no_grad():
            for features, targets, existence_targets, metadata in validation_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                existence_targets = existence_targets.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                mask_logits, existence_logits, iou_logits = parallel_model(
                    features, metadata
                )
                positive = existence_targets > 0
                if positive.any():
                    predictions = mask_logits[positive].sigmoid() >= 0.50
                    truth = targets[positive] >= 0.50
                    intersection = (predictions & truth).sum(dim=(1, 2, 3)).float()
                    denominator = predictions.sum(dim=(1, 2, 3)) + truth.sum(
                        dim=(1, 2, 3)
                    )
                    dice_sum += float(
                        ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum()
                    )
                    targets_iou = dynamic_mask_iou_targets(
                        mask_logits[positive], targets[positive]
                    )
                    iou_error += float(
                        (iou_logits[positive].sigmoid() - targets_iou).abs().sum()
                    )
                    positive_count += int(positive.sum())
                existence_error += float(
                    (existence_logits.sigmoid() - existence_targets).abs().sum()
                )
                sample_count += len(features)
        validation_dice = dice_sum / max(positive_count, 1)
        existence_mae = existence_error / max(sample_count, 1)
        iou_mae = iou_error / max(positive_count, 1)
        validation_score = (
            validation_dice + 0.10 * (1.0 - existence_mae) + 0.10 * (1.0 - iou_mae)
        )
        row = {
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "validation_dice": validation_dice,
            "existence_mae": existence_mae,
            "iou_mae": iou_mae,
            "selection_score": validation_score,
            "epoch_minutes": (time.monotonic() - epoch_started) / 60,
        }
        history.append(row)
        print(
            json.dumps({"event": "proposal_v3_epoch", **row}, separators=(",", ":")),
            flush=True,
        )
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(fold_dir / "history.csv", index=False)
    del parallel_model, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint, history_frame


@torch.no_grad()
def proposal_v3_probability_records(
    image: np.ndarray,
    seeds: Sequence[np.ndarray],
    detector_scores: Sequence[float],
    model: ProposalRefinerV3,
    device: torch.device,
    cfg: CFG,
) -> list[dict[str, Any]]:
    if not seeds:
        return []
    size = (cfg.refiner_crop,) * 2
    bounds: list[tuple[int, int, int, int]] = []
    feature_arrays, metadata_arrays = [], []
    for seed, detector_score in zip(seeds, detector_scores):
        x0, y0, x1, y1 = square_bounds(
            seed, cfg.refiner_context, cfg.refiner_min_crop
        )
        bounds.append((x0, y0, x1, y1))
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1].astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        feature_arrays.append(
            np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32)
            / 255.0
        )
        metadata_arrays.append(proposal_v3_metadata(image, seed, detector_score))
    tensor = torch.from_numpy(np.stack(feature_arrays)).to(device)
    metadata = torch.from_numpy(np.stack(metadata_arrays)).to(device)
    transforms = [lambda value: value, lambda value: value.flip(-1)]
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    transformed_metadata = metadata.repeat(len(transforms), 1)
    mask_chunks, existence_chunks, iou_chunks = [], [], []
    for start in range(0, len(transformed), 8):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mask_logits, existence_logits, iou_logits = model(
                transformed[start : start + 8],
                transformed_metadata[start : start + 8],
            )
        mask_chunks.append(mask_logits.sigmoid().float().cpu())
        existence_chunks.append(existence_logits.sigmoid().float().cpu())
        iou_chunks.append(iou_logits.sigmoid().float().cpu())
    mask_probabilities = torch.cat(mask_chunks)
    existence_probabilities = torch.cat(existence_chunks)
    iou_probabilities = torch.cat(iou_chunks)
    count = len(seeds)
    records: list[dict[str, Any]] = []
    for index, (seed, detector_score, crop_bounds) in enumerate(
        zip(seeds, detector_scores, bounds)
    ):
        views = [
            transforms[view](
                mask_probabilities[view * count + index : view * count + index + 1]
            )[0, 0].numpy()
            for view in range(len(transforms))
        ]
        existence = float(
            np.mean(
                [existence_probabilities[view * count + index].item() for view in range(len(transforms))]
            )
        )
        predicted_iou = float(
            np.mean(
                [iou_probabilities[view * count + index].item() for view in range(len(transforms))]
            )
        )
        x0, y0, x1, y1 = crop_bounds
        records.append(
            {
                "seed": seed,
                "detector_score": float(detector_score),
                "existence": existence,
                "predicted_iou": predicted_iou,
                "bounds": crop_bounds,
                "probability": cv2.resize(
                    np.mean(views, axis=0),
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR,
                ),
            }
        )
    return records


def predictions_for_v3_setting(
    records: Sequence[Mapping[str, Any]],
    setting: ProposalV3Setting,
    minimum_area: int,
) -> tuple[list[np.ndarray], list[float]]:
    candidates: list[tuple[np.ndarray, float]] = []
    for record in records:
        existence = float(record["existence"])
        predicted_iou = float(record["predicted_iou"])
        expected_quality = existence * predicted_iou
        if existence < setting.existence_threshold or expected_quality < setting.score_threshold:
            continue
        crop_mask = record["probability"] >= setting.mask_threshold
        if int(crop_mask.sum()) < minimum_area:
            continue
        full_mask = np.zeros_like(record["seed"], dtype=bool)
        x0, y0, x1, y1 = record["bounds"]
        full_mask[y0:y1, x0:x1] = crop_mask
        foreground_quality = float(record["probability"][crop_mask].mean())
        score = (
            max(float(record["detector_score"]), 1e-3) ** 0.35
            * expected_quality
            * foreground_quality
        )
        candidates.append((full_mask, score))
    if not candidates and records:
        record = max(
            records,
            key=lambda value: float(value["existence"]) * float(value["predicted_iou"]),
        )
        crop_mask = record["probability"] >= setting.mask_threshold
        if crop_mask.any():
            full_mask = np.zeros_like(record["seed"], dtype=bool)
            x0, y0, x1, y1 = record["bounds"]
            full_mask[y0:y1, x0:x1] = crop_mask
            candidates = [
                (
                    full_mask,
                    0.001
                    * float(record["existence"])
                    * float(record["predicted_iou"]),
                )
            ]
    if not candidates:
        return [], []
    masks, scores = zip(*candidates)
    return resolve_overlaps(list(masks), list(scores), minimum_area)


def proposal_v3_settings() -> tuple[ProposalV3Setting, ...]:
    return tuple(
        ProposalV3Setting(existence, quality, mask)
        for existence in (0.25, 0.40, 0.55)
        for quality in (0.20, 0.30, 0.40)
        for mask in (0.45, 0.50, 0.55)
    )


def parse_proposal_v3_setting(name: str) -> ProposalV3Setting:
    existence_text, quality_text, mask_text = name.split("_")
    return ProposalV3Setting(
        float(existence_text.removeprefix("e")),
        float(quality_text.removeprefix("q")),
        float(mask_text.removeprefix("m")),
    )


def summary_with_sq(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    summary = official_summary(rows)
    summary["sq"] = (
        summary["sum_iou"] / summary["tp"] if summary["tp"] else 0.0
    )
    return summary


def v2_samples_from_v3_catalog(
    samples: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    eligible = [value for value in samples if float(value["detector_score"]) >= 0.18]
    if not eligible:
        eligible = list(samples)
    return sorted(
        eligible, key=lambda value: float(value["detector_score"]), reverse=True
    )[:20]


def evaluate_proposal_refiner_v3_oof(
    fold: int,
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    indices: Mapping[str, Any],
    v3_checkpoint: Path,
    v2_checkpoint: Path,
    cfg: CFG,
    settings: Sequence[ProposalV3Setting],
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    v3 = ProposalRefinerV3()
    v3.load_state_dict(torch.load(v3_checkpoint, map_location=device), strict=True)
    v3.to(device).eval()
    v2 = ProposalRefinerV2()
    v2.load_state_dict(torch.load(v2_checkpoint, map_location=device), strict=True)
    v2.to(device).eval()
    grouped = catalog_by_filename(validation_samples)
    setting_rows: dict[str, list[dict[str, Any]]] = {
        setting.name: [] for setting in settings
    }
    v2_rows: list[dict[str, Any]] = []
    v2_setting = ProposalV2Setting(0.50, 0.50)
    v2_cfg = legacy_inference_cfg()
    v2_cfg.refiner_crop = 384
    v2_cfg.refiner_min_crop = 128
    v2_cfg.minimum_area = cfg.minimum_area
    for number, (filename, samples) in enumerate(sorted(grouped.items()), 1):
        image = cv2.imread(str(image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / filename)
        seeds = [mask_utils.decode(sample["seed_rle"]).astype(bool) for sample in samples]
        detector_scores = [float(sample["detector_score"]) for sample in samples]
        v3_records = proposal_v3_probability_records(
            image, seeds, detector_scores, v3, device, cfg
        )
        selected_v2 = v2_samples_from_v3_catalog(samples)
        v2_seeds = [
            mask_utils.decode(sample["seed_rle"]).astype(bool)
            for sample in selected_v2
        ]
        v2_scores = [float(sample["detector_score"]) for sample in selected_v2]
        v2_records = proposal_v2_probability_records(
            image, v2_seeds, v2_scores, v2, device, v2_cfg
        )
        v2_masks, _ = predictions_for_proposal_setting(
            v2_records, v2_setting, cfg.minimum_area
        )
        ground_truth = ground_truth_sets(filename, indices)
        for annotator, gt_masks in enumerate(ground_truth):
            v2_rows.append(
                audit_metric_row(
                    filename, fold, annotator, "proposal_v2", gt_masks, v2_masks
                )
            )
        for setting in settings:
            masks, _ = predictions_for_v3_setting(
                v3_records, setting, cfg.minimum_area
            )
            for annotator, gt_masks in enumerate(ground_truth):
                setting_rows[setting.name].append(
                    audit_metric_row(
                        filename, fold, annotator, setting.name, gt_masks, masks
                    )
                )
        if number % 20 == 0 or number == len(grouped):
            print(
                {
                    "stage": "proposal_v3_oof",
                    "fold": fold,
                    "image": f"{number}/{len(grouped)}",
                },
                flush=True,
            )
    summaries = {
        name: summary_with_sq(rows) for name, rows in setting_rows.items()
    }
    best_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["official_pq"],
            summaries[name]["sq"],
            -summaries[name]["mean_predictions"],
        ),
    )
    fold_dir = output_dir / "oof" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(v2_rows).to_csv(
        fold_dir / "proposal_v2_by_annotator.csv", index=False
    )
    pd.DataFrame(setting_rows[best_name]).to_csv(
        fold_dir / "proposal_v3_best_by_annotator.csv", index=False
    )
    baseline = summary_with_sq(v2_rows)
    report = {
        "fold": fold,
        "proposal_v2": baseline,
        "settings": summaries,
        "best_setting": best_name,
        "best": summaries[best_name],
        "official_pq_delta": summaries[best_name]["official_pq"] - baseline["official_pq"],
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    del v3, v2
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {"event": "proposal_v3_fold_complete", **report}, separators=(",", ":")
        ),
        flush=True,
    )
    return report


@torch.no_grad()
def predict_proposal_v3_instances(
    image_path: Path,
    detector: Any,
    model: ProposalRefinerV3,
    device: torch.device,
    cfg: CFG,
    setting: ProposalV3Setting,
) -> tuple[list[np.ndarray], list[float]]:
    image, seeds, scores = raw_legacy_candidates(image_path, detector, device, cfg)
    if not seeds:
        seeds = [classical_fallback_seed(image, cfg.minimum_area)]
        scores = [0.01]
    selected = selected_v3_proposal_indices(seeds, scores, cfg)
    chosen_seeds = [seeds[index] for index in selected]
    chosen_scores = [float(scores[index]) for index in selected]
    records = proposal_v3_probability_records(
        image, chosen_seeds, chosen_scores, model, device, cfg
    )
    return predictions_for_v3_setting(records, setting, cfg.minimum_area)


def fuse_v3_weighted(
    predictions: Sequence[tuple[int, np.ndarray, float]], minimum_area: int
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    policy = FusionPolicy(
        "s3_weighted",
        match_iou=0.20,
        minimum_support=3,
        mask_mode="majority",
        minimum_area=minimum_area,
    )
    clusters = cluster_fold_predictions(predictions, policy.match_iou)
    support_histogram: dict[int, int] = {}
    masks, scores, supports = [], [], []
    for cluster in clusters:
        support = len(cluster)
        support_histogram[support] = support_histogram.get(support, 0) + 1
        if support < policy.minimum_support:
            continue
        weights = np.asarray([max(float(item[2]), 1e-4) for item in cluster])
        weighted_votes = np.zeros_like(cluster[0][1], dtype=np.float32)
        for weight, item in zip(weights, cluster):
            weighted_votes += float(weight) * item[1].astype(np.float32)
        mask = weighted_votes >= (0.50 * float(weights.sum()))
        if mask.sum() < minimum_area:
            mask = cluster_representative(cluster)[1].copy()
        if mask.sum() >= minimum_area:
            masks.append(mask)
            scores.append(float(weights.mean()))
            supports.append(support)
    used_fallback = False
    if not masks and clusters:
        fallback = max(
            clusters,
            key=lambda value: (len(value), np.mean([item[2] for item in value])),
        )
        masks = [cluster_representative(fallback)[1].copy()]
        scores = [float(np.mean([item[2] for item in fallback]))]
        supports = [len(fallback)]
        used_fallback = True
    masks, scores, supports = resolve_fused_overlaps(
        masks, scores, supports, policy
    )
    return masks, scores, {
        "clusters": len(clusters),
        "support_histogram": support_histogram,
        "accepted": len(masks),
        "accepted_support": supports,
        "fallback": used_fallback,
    }


def validate_submission_frame(
    submission: pd.DataFrame, test_files: Sequence[Path], label: str
) -> None:
    test_ids = {path.stem for path in test_files}
    submitted_ids = set(submission["filament_id"].str.rsplit("_", n=1).str[0])
    if submitted_ids != test_ids:
        raise RuntimeError(f"{label} misses {len(test_ids - submitted_ids)} test images")
    if submission.empty or not submission["filament_id"].is_unique:
        raise RuntimeError(f"Invalid {label} candidate")


def run_proposal_v3_test(
    paths: Mapping[str, Path],
    baseline_root: Path,
    output_dir: Path,
    cfg: CFG,
    setting: ProposalV3Setting,
) -> dict[str, Any]:
    from ultralytics import YOLO

    models = []
    for fold in range(cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(str(baseline_root / "models" / f"fold_{fold}" / "detector.pt"))
        model = ProposalRefinerV3()
        model.load_state_dict(
            torch.load(
                output_dir / "models" / f"fold_{fold}" / "proposal_refiner_v3.pt",
                map_location=device,
            ),
            strict=True,
        )
        models.append((fold, detector, model.to(device).eval(), device))
    medoid_policy = FusionPolicy(
        "s3_medoid",
        match_iou=0.20,
        minimum_support=3,
        mask_mode="medoid",
        minimum_area=cfg.minimum_area,
    )
    medoid_rows: list[dict[str, str]] = []
    weighted_rows: list[dict[str, str]] = []
    diagnostics = []
    test_files = sorted(paths["test"].glob("*.jpeg"))
    for number, image_path in enumerate(test_files, 1):
        predictions = []
        for fold, detector, model, device in models:
            masks, scores = predict_proposal_v3_instances(
                image_path, detector, model, device, cfg, setting
            )
            predictions.extend(
                (fold, mask, score) for mask, score in zip(masks, scores)
            )
        medoid_masks, _, medoid_detail = fuse_predictions_with_policy(
            predictions, medoid_policy
        )
        weighted_masks, _, weighted_detail = fuse_v3_weighted(
            predictions, cfg.minimum_area
        )
        for rows, masks in (
            (medoid_rows, medoid_masks), (weighted_rows, weighted_masks)
        ):
            for instance_index, mask in enumerate(masks, 1):
                rows.append(
                    {
                        "filament_id": f"{image_path.stem}_{instance_index}",
                        "segmentation_rle": encode_mask(mask),
                    }
                )
        diagnostics.append(
            {
                "filename": image_path.name,
                "raw_instances": len(predictions),
                "medoid_instances": len(medoid_masks),
                "weighted_instances": len(weighted_masks),
                "medoid_fallback": bool(medoid_detail["fallback"]),
                "weighted_fallback": bool(weighted_detail["fallback"]),
                "support_histogram": json.dumps(
                    medoid_detail["support_histogram"],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        if number % 10 == 0 or number == len(test_files):
            print(
                {
                    "stage": "proposal_v3_test",
                    "image": f"{number}/{len(test_files)}",
                    "medoid_rows": len(medoid_rows),
                    "weighted_rows": len(weighted_rows),
                },
                flush=True,
            )
    outputs = {}
    for name, rows in (("s3_medoid", medoid_rows), ("s3_weighted", weighted_rows)):
        submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        validate_submission_frame(submission, test_files, f"proposal v3 {name}")
        path = output_dir / f"candidate_proposal_v3_{name}.csv"
        submission.to_csv(path, index=False)
        outputs[name] = {
            "path": str(path),
            "rows": len(submission),
            "images": len(test_files),
            "mean_instances": len(submission) / len(test_files),
        }
    pd.DataFrame(diagnostics).to_csv(
        output_dir / "test_diagnostics.csv", index=False
    )
    return {
        "candidates": outputs,
        "fallback_images": {
            "s3_medoid": int(sum(row["medoid_fallback"] for row in diagnostics)),
            "s3_weighted": int(sum(row["weighted_fallback"] for row in diagnostics)),
        },
    }


def discover_proposal_v3_full_root() -> Path | None:
    """Locate the completed five-fold V3 checkpoints attached as a kernel source."""
    candidates = sorted(
        {
            path.parents[2]
            for path in Path("/kaggle/input").glob(
                "**/proposal_refiner_v3_full/models/fold_0/proposal_refiner_v3.pt"
            )
        }
    )
    valid = [
        root
        for root in candidates
        if all(
            (root / "models" / f"fold_{fold}" / "proposal_refiner_v3.pt").is_file()
            for fold in range(5)
        )
    ]
    if len(valid) > 1:
        raise RuntimeError(f"Multiple full proposal v3 roots found: {valid}")
    return valid[0] if valid else None


@torch.no_grad()
def run_proposal_v23_hybrid_sweep(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v2_root: Path,
    v3_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Build a compact V2/V3 architecture-by-fusion test matrix."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    v2_cfg = legacy_inference_cfg()
    v2_cfg.refiner_crop = 384
    v2_cfg.refiner_min_crop = 128
    v2_cfg.seed_nms_iou = 0.45
    v2_cfg.minimum_area = 96
    v3_cfg = legacy_inference_cfg()
    v3_cfg.detector_conf = 0.05
    v3_cfg.refiner_crop = 448
    v3_cfg.refiner_min_crop = 128
    v3_cfg.refiner_batch = 4
    v3_cfg.seed_nms_iou = 0.45
    v3_cfg.minimum_area = 96
    v2_setting = ProposalV2Setting(quality_threshold=0.50, mask_threshold=0.50)
    v3_setting = ProposalV3Setting(
        existence_threshold=0.55,
        score_threshold=0.30,
        mask_threshold=0.50,
    )
    policies = {
        "medoid": FusionPolicy(
            "s3_medoid", 0.20, 3, "medoid", minimum_area=96
        ),
        "majority": FusionPolicy(
            "s3_majority", 0.20, 3, "majority", minimum_area=96
        ),
    }
    variants = {
        "hybrid012_s3_medoid": ({0, 1, 2}, policies["medoid"]),
        "hybrid012_s3_majority": ({0, 1, 2}, policies["majority"]),
        "v3all_s3_medoid": ({0, 1, 2, 3, 4}, policies["medoid"]),
        "v3all_s3_majority": ({0, 1, 2, 3, 4}, policies["majority"]),
    }
    models = []
    for fold in range(v2_cfg.n_folds):
        device = fold_device(fold)
        detector = YOLO(
            str(baseline_root / "models" / f"fold_{fold}" / "detector.pt")
        )
        v2_model = ProposalRefinerV2()
        v2_model.load_state_dict(
            torch.load(
                v2_root / "models" / f"fold_{fold}" / "proposal_refiner_v2.pt",
                map_location=device,
            ),
            strict=True,
        )
        v3_model = ProposalRefinerV3()
        v3_model.load_state_dict(
            torch.load(
                v3_root / "models" / f"fold_{fold}" / "proposal_refiner_v3.pt",
                map_location=device,
            ),
            strict=True,
        )
        models.append(
            (
                fold,
                detector,
                v2_model.to(device).eval(),
                v3_model.to(device).eval(),
                device,
            )
        )
    rows_by_variant: dict[str, list[dict[str, str]]] = {
        name: [] for name in variants
    }
    diagnostics_by_variant: dict[str, list[dict[str, Any]]] = {
        name: [] for name in variants
    }
    test_files = sorted(paths["test"].glob("*.jpeg"))
    if len(test_files) != 180:
        raise RuntimeError(f"Expected 180 test images, found {len(test_files)}")
    for number, image_path in enumerate(test_files, 1):
        by_architecture: dict[str, list[tuple[int, np.ndarray, float]]] = {
            "v2": [],
            "v3": [],
        }
        for fold, detector, v2_model, v3_model, device in models:
            v2_masks, v2_scores = predict_proposal_v2_instances(
                image_path, detector, v2_model, device, v2_cfg, v2_setting
            )
            v3_masks, v3_scores = predict_proposal_v3_instances(
                image_path, detector, v3_model, device, v3_cfg, v3_setting
            )
            by_architecture["v2"].extend(
                (fold, mask, score) for mask, score in zip(v2_masks, v2_scores)
            )
            by_architecture["v3"].extend(
                (fold, mask, score) for mask, score in zip(v3_masks, v3_scores)
            )
        for name, (v3_folds, policy) in variants.items():
            predictions = [
                item
                for architecture, values in by_architecture.items()
                for item in values
                if (item[0] in v3_folds) == (architecture == "v3")
            ]
            masks, _, detail = fuse_predictions_with_policy(predictions, policy)
            for instance_index, mask in enumerate(masks, 1):
                rows_by_variant[name].append(
                    {
                        "filament_id": f"{image_path.stem}_{instance_index}",
                        "segmentation_rle": encode_mask(mask),
                    }
                )
            diagnostics_by_variant[name].append(
                {
                    "filename": image_path.name,
                    "raw_instances": len(predictions),
                    "instances": len(masks),
                    "mean_area": float(np.mean([mask.sum() for mask in masks])),
                    "fallback": bool(detail["fallback"]),
                    "support_histogram": json.dumps(
                        detail["support_histogram"],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            )
        if number % 10 == 0 or number == len(test_files):
            print(
                {
                    "stage": "proposal_v23_hybrid_sweep",
                    "image": f"{number}/{len(test_files)}",
                    "rows": {
                        name: len(rows) for name, rows in rows_by_variant.items()
                    },
                },
                flush=True,
            )
    output_variants: dict[str, Any] = {}
    for name, rows in rows_by_variant.items():
        submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        validate_submission_frame(submission, test_files, name)
        submission_path = output_dir / f"submission_{name}.csv"
        diagnostics_path = output_dir / f"diagnostics_{name}.csv"
        submission.to_csv(submission_path, index=False)
        pd.DataFrame(diagnostics_by_variant[name]).to_csv(
            diagnostics_path, index=False
        )
        output_variants[name] = {
            "path": str(submission_path),
            "sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
            "rows": len(submission),
            "images": len(test_files),
            "mean_instances": len(submission) / len(test_files),
            "fallback_images": int(
                sum(row["fallback"] for row in diagnostics_by_variant[name])
            ),
        }
    if len({value["sha256"] for value in output_variants.values()}) != 4:
        raise RuntimeError("The hybrid sweep produced duplicate submissions")
    report = {
        "status": "four_candidates_ready",
        "architectures": {
            "hybrid012": "V3 folds 0-2 and V2 folds 3-4",
            "v3all": "V3 folds 0-4",
        },
        "strict_oof_reference": {
            "v2_all_s3_medoid": 0.4311884919628038,
            "hybrid012_s3_medoid": 0.43520179088181965,
            "v3_all_s3_medoid": 0.4343910743782759,
        },
        "settings": {"v2": asdict(v2_setting), "v3": asdict(v3_setting)},
        "variants": output_variants,
        "runtime_hours": (time.monotonic() - run_started) / 3600,
        "submitted": False,
    }
    (output_dir / "hybrid_sweep_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "proposal_v23_hybrid_complete", **report},
            separators=(",", ":"),
        ),
        flush=True,
    )
    return report


def run_proposal_refiner_v3(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v2_root: Path,
    output_dir: Path,
    run_started: float,
    resume_root: Path | None = None,
) -> dict[str, Any]:
    """Train V3 strictly OOF, optionally reusing a completed fold-0 checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(baseline_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.detector_conf = 0.05
    cfg.seed_nms_iou = 0.45
    cfg.refiner_crop = 448
    cfg.refiner_min_crop = 128
    cfg.refiner_batch = 4
    # The original six-epoch fold 0 is reused by the resume run. Four epochs
    # for folds 1-4 preserve its steadily improving validation regime while
    # leaving enough time for exact OOF evaluation and five-fold test inference.
    cfg.refiner_epochs = 4 if resume_root is not None else 6
    cfg.minimum_area = 96
    cfg.num_workers = 4
    catalog = build_v3_oof_catalog(
        manifest, paths["train"], baseline_root, medoids, indices, cfg
    )
    catalog_report = {
        "samples": len(catalog),
        "matched": int(sum(float(value["existence"]) > 0 for value in catalog)),
        "negatives": int(sum(float(value["existence"]) == 0 for value in catalog)),
        "mean_existence_matched": float(
            np.mean(
                [
                    float(value["existence"])
                    for value in catalog
                    if float(value["existence"]) > 0
                ]
            )
        ),
        "strictly_oof": True,
    }
    (output_dir / "catalog_report.json").write_text(
        json.dumps(catalog_report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "proposal_v3_catalog_complete", **catalog_report},
            separators=(",", ":"),
        ),
        flush=True,
    )
    all_settings = proposal_v3_settings()
    selected_setting: ProposalV3Setting | None = None
    fold_reports: dict[str, Any] = {}
    oof_rows: list[dict[str, Any]] = []
    v2_oof_rows: list[dict[str, Any]] = []
    start_fold = 0
    if resume_root is not None:
        source_fold = resume_root / "oof" / "fold_0"
        source_model = resume_root / "models" / "fold_0"
        required = (
            source_fold / "report.json",
            source_fold / "proposal_v2_by_annotator.csv",
            source_fold / "proposal_v3_best_by_annotator.csv",
            source_model / "proposal_refiner_v3.pt",
            source_model / "history.csv",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete V3 fold-0 resume source: {missing}")
        destination_fold = output_dir / "oof" / "fold_0"
        destination_model = output_dir / "models" / "fold_0"
        destination_fold.mkdir(parents=True, exist_ok=True)
        destination_model.mkdir(parents=True, exist_ok=True)
        for source in required[:3]:
            shutil.copy2(source, destination_fold / source.name)
        for source in required[3:]:
            shutil.copy2(source, destination_model / source.name)
        fold0_report = json.loads((source_fold / "report.json").read_text())
        fold0_history = pd.read_csv(source_model / "history.csv")
        fold_reports["0"] = {
            **fold0_report,
            "best_crop_dice": float(fold0_history["validation_dice"].max()),
            "runtime_hours": 0.0,
            "reused_checkpoint": True,
        }
        selected_setting = parse_proposal_v3_setting(fold0_report["best_setting"])
        oof_rows.extend(
            pd.read_csv(source_fold / "proposal_v3_best_by_annotator.csv").to_dict(
                "records"
            )
        )
        v2_oof_rows.extend(
            pd.read_csv(source_fold / "proposal_v2_by_annotator.csv").to_dict(
                "records"
            )
        )
        start_fold = 1
        print(
            json.dumps(
                {
                    "event": "proposal_v3_fold0_resumed",
                    "source": str(resume_root),
                    "selected_setting": selected_setting.name,
                    "new_fold_epochs": cfg.refiner_epochs,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    for fold in range(start_fold, cfg.n_folds):
        train_files = set(
            manifest.loc[manifest["fold"] != fold, "filename"].astype(str)
        )
        validation_files = set(
            manifest.loc[manifest["fold"] == fold, "filename"].astype(str)
        )
        train_samples = balanced_v3_training_samples(
            [value for value in catalog if value["filename"] in train_files]
        )
        validation_samples = [
            value for value in catalog if value["filename"] in validation_files
        ]
        checkpoint, history = train_proposal_refiner_v3(
            fold,
            train_samples,
            validation_samples,
            paths["train"],
            v2_root / "models" / f"fold_{fold}" / "proposal_refiner_v2.pt",
            cfg,
            output_dir,
        )
        settings = all_settings if selected_setting is None else (selected_setting,)
        report = evaluate_proposal_refiner_v3_oof(
            fold,
            validation_samples,
            paths["train"],
            indices,
            checkpoint,
            v2_root / "models" / f"fold_{fold}" / "proposal_refiner_v2.pt",
            cfg,
            settings,
            output_dir,
        )
        fold_reports[str(fold)] = {
            **report,
            "best_crop_dice": float(history["validation_dice"].max()),
            "runtime_hours": (time.monotonic() - run_started) / 3600,
        }
        if selected_setting is None:
            selected_setting = parse_proposal_v3_setting(report["best_setting"])
            if (
                report["best"]["official_pq"] < 0.448
                or report["official_pq_delta"] < 0.010
                or report["best"]["rq"] + 0.003 < report["proposal_v2"]["rq"]
            ):
                result = {
                    "status": "rejected_after_fold0",
                    "fold0_pq_gate": 0.448,
                    "fold0_delta_gate": 0.010,
                    "catalog": catalog_report,
                    "folds": fold_reports,
                    "selected_setting": asdict(selected_setting),
                    "submitted": False,
                }
                (output_dir / "proposal_v3_report.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {"event": "proposal_v3_rejected", **result},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return result
        best_path = (
            output_dir / "oof" / f"fold_{fold}" / "proposal_v3_best_by_annotator.csv"
        )
        oof_rows.extend(pd.read_csv(best_path).to_dict("records"))
        v2_oof_rows.extend(
            pd.read_csv(
                output_dir
                / "oof"
                / f"fold_{fold}"
                / "proposal_v2_by_annotator.csv"
            ).to_dict("records")
        )
        elapsed_hours = (time.monotonic() - run_started) / 3600
        if elapsed_hours > 5.40 and fold < cfg.n_folds - 1:
            result = {
                "status": "checkpoints_partial_runtime_guard",
                "runtime_hours": elapsed_hours,
                "catalog": catalog_report,
                "folds": fold_reports,
                "selected_setting": asdict(selected_setting),
                "submitted": False,
            }
            (output_dir / "proposal_v3_report.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {"event": "proposal_v3_runtime_guard", **result},
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return result
    aggregate = summary_with_sq(oof_rows)
    paired_v2 = summary_with_sq(v2_oof_rows)
    official_pq_delta = aggregate["official_pq"] - paired_v2["official_pq"]
    improved_folds = sum(
        value["best"]["official_pq"] > value["proposal_v2"]["official_pq"]
        for value in fold_reports.values()
    )
    release_gate = {
        "official_pq_delta": 0.005,
        "minimum_improved_folds": 3,
    }
    if (
        official_pq_delta < release_gate["official_pq_delta"]
        or improved_folds < release_gate["minimum_improved_folds"]
    ):
        result = {
            "status": "rejected_after_full_oof",
            "release_gate": release_gate,
            "improved_folds": improved_folds,
            "aggregate_oof": aggregate,
            "paired_v2_oof": paired_v2,
            "official_pq_delta": official_pq_delta,
            "catalog": catalog_report,
            "folds": fold_reports,
            "selected_setting": asdict(selected_setting),
            "submitted": False,
        }
        (output_dir / "proposal_v3_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {"event": "proposal_v3_full_rejected", **result},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return result
    elapsed_hours = (time.monotonic() - run_started) / 3600
    if elapsed_hours > 5.85:
        result = {
            "status": "checkpoints_complete_runtime_guard",
            "runtime_hours": elapsed_hours,
            "aggregate_oof": aggregate,
            "paired_v2_oof": paired_v2,
            "official_pq_delta": official_pq_delta,
            "catalog": catalog_report,
            "folds": fold_reports,
            "selected_setting": asdict(selected_setting),
            "submitted": False,
        }
        (output_dir / "proposal_v3_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {"event": "proposal_v3_runtime_guard", **result},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return result
    test = run_proposal_v3_test(
        paths, baseline_root, output_dir, cfg, selected_setting
    )
    result = {
        "status": "candidates_ready_no_submission",
        "release_gate": release_gate,
        "improved_folds": improved_folds,
        "aggregate_oof": aggregate,
        "paired_v2_oof": paired_v2,
        "official_pq_delta": official_pq_delta,
        "catalog": catalog_report,
        "folds": fold_reports,
        "selected_setting": asdict(selected_setting),
        "test": test,
        "runtime_hours": (time.monotonic() - run_started) / 3600,
        "submitted": False,
    }
    (output_dir / "proposal_v3_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "proposal_v3_complete", **result}, separators=(",", ":")
        ),
        flush=True,
    )
    return result


@dataclass(frozen=True)
class GeometryV4Setting:
    geometry_weight: float
    mask_threshold: float

    @property
    def name(self) -> str:
        return f"a{self.geometry_weight:.2f}_m{self.mask_threshold:.2f}"


def discover_proposal_v3_root() -> Path | None:
    """Locate the fold-0 V3 checkpoint used as the frozen selector."""
    candidates = sorted(
        {
            path.parents[2]
            for path in Path("/kaggle/input").glob(
                "**/proposal_refiner_v3/models/fold_0/proposal_refiner_v3.pt"
            )
        }
    )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple proposal v3 roots found: {candidates}")
    return candidates[0] if candidates else None


class GeometryRefinerV4(nn.Module):
    """High-resolution mask fine-tuning of the already proven V3 refiner."""

    def __init__(self, v3_checkpoint: Path) -> None:
        super().__init__()
        self.refiner = ProposalRefinerV3()
        self.refiner.load_state_dict(
            torch.load(v3_checkpoint, map_location="cpu"), strict=True
        )
        for module in (
            self.refiner.quality_head,
            self.refiner.metadata_project,
            self.refiner.existence_correction,
            self.refiner.iou_head,
        ):
            module.requires_grad_(False)

    def forward(
        self, tensor: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_logits, _, predicted_iou = self.refiner(tensor, metadata)
        return mask_logits, predicted_iou


def smoke_check_geometry_refiner_v4(
    v3_checkpoint: Path,
    image_dir: Path,
    indices: Mapping[str, Any],
    medoids: Mapping[str, str],
    cfg: CFG,
) -> None:
    """Run a real annotated crop through dual-GPU forward/backward."""
    smoke_sample: dict[str, Any] | None = None
    for filename in sorted(medoids):
        image_id = medoids[filename]
        info = indices["images_by_id"][image_id]
        annotations = [
            value
            for value in indices["annotations_by_image"].get(image_id, [])
            if not value.get("iscrowd", 0)
        ]
        if not annotations:
            continue
        mask = annotation_to_mask(
            annotations[0], int(info["height"]), int(info["width"])
        )
        smoke_sample = {
            "filename": filename,
            "seed_rle": binary_rle(mask),
            "target_rles": [binary_rle(mask)],
            "existence": 1.0,
            "detector_score": 1.0,
        }
        break
    if smoke_sample is None:
        raise RuntimeError("No annotated instance available for geometry smoke test")
    dataset = ProposalRefinerV3Dataset([smoke_sample], image_dir, cfg, True)
    feature, target, _, metadata = dataset[0]
    device = torch.device("cuda:0")
    gpu_count = torch.cuda.device_count()
    model = GeometryRefinerV4(v3_checkpoint).to(device)
    parallel_model: nn.Module = (
        nn.DataParallel(model, device_ids=list(range(gpu_count)))
        if gpu_count > 1
        else model
    )
    effective_batch = cfg.refiner_batch * max(gpu_count, 1)
    features = feature.unsqueeze(0).repeat(effective_batch, 1, 1, 1).to(device)
    targets = target.unsqueeze(0).repeat(effective_batch, 1, 1, 1).to(device)
    metadata_batch = metadata.unsqueeze(0).repeat(effective_batch, 1).to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, predicted_iou = parallel_model(features, metadata_batch)
        boundary_bce, dice = proposal_mask_losses(logits, targets)
        topology = soft_cldice_losses(logits.sigmoid(), targets)
        loss = (0.42 * boundary_bce + 0.48 * dice + 0.10 * topology).mean()
    loss.backward()
    if logits.shape != targets.shape or predicted_iou.shape != (effective_batch,):
        raise RuntimeError(
            f"Geometry V4 smoke shapes invalid: {logits.shape}, {predicted_iou.shape}"
        )
    print(
        {
            "stage": "geometry_v4_smoke_passed",
            "gpus": gpu_count,
            "batch": effective_batch,
            "crop": cfg.refiner_crop,
            "architecture": "v3_resnet18_highres",
            "filename": smoke_sample["filename"],
            "loss": float(loss.detach()),
        },
        flush=True,
    )
    del parallel_model, model, features, targets, metadata_batch, logits, predicted_iou, loss
    gc.collect()
    torch.cuda.empty_cache()


def geometry_v4_settings() -> tuple[GeometryV4Setting, ...]:
    return tuple(
        GeometryV4Setting(alpha, threshold)
        for alpha in (0.25, 0.50, 0.75, 1.00)
        for threshold in (0.38, 0.42, 0.46, 0.50, 0.54, 0.58)
    )


def train_geometry_refiner_v4(
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    v3_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """Train only the fold-0 geometry experiment on matched OOF proposals."""
    fold_dir = output_dir / "models" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "geometry_refiner_v4.pt"
    positive_train = [value for value in train_samples if float(value["existence"]) > 0]
    positive_validation = [
        value for value in validation_samples if float(value["existence"]) > 0
    ]
    train_dataset = ProposalRefinerV3Dataset(positive_train, image_dir, cfg, True)
    validation_dataset = ProposalRefinerV3Dataset(
        positive_validation, image_dir, cfg, False
    )
    gpu_count = torch.cuda.device_count()
    effective_batch = cfg.refiner_batch * max(gpu_count, 1)
    loader_options = {
        "num_workers": min(cfg.num_workers, 4),
        "pin_memory": True,
        "persistent_workers": min(cfg.num_workers, 4) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 400),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=effective_batch * 2,
        shuffle=False,
        **loader_options,
    )
    device = torch.device("cuda:0")
    model = GeometryRefinerV4(v3_checkpoint).to(device)
    parallel_model: nn.Module = (
        nn.DataParallel(model, device_ids=list(range(gpu_count)))
        if gpu_count > 1
        else model
    )
    encoder_modules = (
        model.refiner.stem,
        model.refiner.layer1,
        model.refiner.layer2,
        model.refiner.layer3,
        model.refiner.layer4,
    )
    backbone_ids = {
        id(parameter)
        for module in encoder_modules
        for parameter in module.parameters()
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) in backbone_ids
                ],
                "lr": 2e-5,
            },
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in backbone_ids
                ],
                "lr": 8e-5,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.refiner_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    print(
        {
            "stage": "geometry_v4_train",
            "fold": 0,
            "architecture": "v3_resnet18_highres",
            "initialization": str(v3_checkpoint),
            "crop": cfg.refiner_crop,
            "gpus": gpu_count,
            "effective_batch": effective_batch,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "selection_frozen": True,
        },
        flush=True,
    )
    for epoch in range(cfg.refiner_epochs):
        epoch_started = time.monotonic()
        parallel_model.train()
        loss_sum = 0.0
        for batch_index, (features, targets, existence_targets, metadata) in enumerate(
            train_loader, 1
        ):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            existence_targets = existence_targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mask_logits, _ = parallel_model(features, metadata)
                boundary_bce, dice = proposal_mask_losses(mask_logits, targets)
                topology = soft_cldice_losses(mask_logits.sigmoid(), targets)
                weights = 0.50 + 0.50 * existence_targets
                mask_loss = (
                    ((0.42 * boundary_bce + 0.48 * dice + 0.10 * topology) * weights).sum()
                    / weights.sum().clamp_min(1.0)
                )
                loss = mask_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    {
                        "stage": "geometry_v4_train_progress",
                        "epoch": epoch + 1,
                        "batch": f"{batch_index}/{len(train_loader)}",
                        "minutes": round((time.monotonic() - epoch_started) / 60, 2),
                        "mean_loss": round(loss_sum / batch_index, 6),
                        "gpu_memory_gb": [
                            round(torch.cuda.memory_allocated(index) / 2**30, 2)
                            for index in range(gpu_count)
                        ],
                    },
                    flush=True,
                )

        parallel_model.eval()
        dice_sum = iou_error = 0.0
        sample_count = 0
        with torch.no_grad():
            for features, targets, _, metadata in validation_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    mask_logits, iou_logits = parallel_model(features, metadata)
                predictions = mask_logits.sigmoid() >= 0.50
                truth = targets >= 0.50
                intersection = (predictions & truth).sum(dim=(1, 2, 3)).float()
                denominator = predictions.sum(dim=(1, 2, 3)) + truth.sum(
                    dim=(1, 2, 3)
                )
                dice_sum += float(
                    ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum()
                )
                iou_targets = dynamic_mask_iou_targets(mask_logits, targets)
                iou_error += float(
                    (iou_logits.sigmoid() - iou_targets).abs().sum()
                )
                sample_count += len(features)
        validation_dice = dice_sum / max(sample_count, 1)
        iou_mae = iou_error / max(sample_count, 1)
        row = {
            "fold": 0,
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "validation_dice": validation_dice,
            "iou_mae": iou_mae,
            "epoch_minutes": (time.monotonic() - epoch_started) / 60,
        }
        history.append(row)
        print(
            json.dumps({"event": "geometry_v4_epoch", **row}, separators=(",", ":")),
            flush=True,
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(fold_dir / "history.csv", index=False)
    del parallel_model, model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint, history_frame


@torch.no_grad()
def geometry_v4_probability_records(
    image: np.ndarray,
    seeds: Sequence[np.ndarray],
    detector_scores: Sequence[float],
    model: GeometryRefinerV4,
    device: torch.device,
    cfg: CFG,
) -> list[dict[str, Any]]:
    if not seeds:
        return []
    size = (cfg.refiner_crop,) * 2
    bounds: list[tuple[int, int, int, int]] = []
    feature_arrays, metadata_arrays = [], []
    for seed, detector_score in zip(seeds, detector_scores):
        x0, y0, x1, y1 = square_bounds(
            seed, cfg.refiner_context, cfg.refiner_min_crop
        )
        bounds.append((x0, y0, x1, y1))
        image_crop = cv2.resize(
            image[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR
        )
        seed_crop = cv2.resize(
            seed[y0:y1, x0:x1].astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        feature_arrays.append(
            np.stack([image_crop, clahe, seed_crop * 255], axis=0).astype(np.float32)
            / 255.0
        )
        metadata_arrays.append(proposal_v3_metadata(image, seed, detector_score))
    tensor = torch.from_numpy(np.stack(feature_arrays)).to(device)
    metadata = torch.from_numpy(np.stack(metadata_arrays)).to(device)
    transforms = [lambda value: value, lambda value: value.flip(-1)]
    transformed = torch.cat([transform(tensor) for transform in transforms], dim=0)
    transformed_metadata = metadata.repeat(len(transforms), 1)
    mask_chunks, iou_chunks = [], []
    for start in range(0, len(transformed), 4):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            mask_logits, iou_logits = model(
                transformed[start : start + 4],
                transformed_metadata[start : start + 4],
            )
        mask_chunks.append(mask_logits.sigmoid().float().cpu())
        iou_chunks.append(iou_logits.sigmoid().float().cpu())
    probabilities = torch.cat(mask_chunks)
    predicted_ious = torch.cat(iou_chunks)
    count = len(seeds)
    records: list[dict[str, Any]] = []
    for index, (seed, detector_score, crop_bounds) in enumerate(
        zip(seeds, detector_scores, bounds)
    ):
        views = [
            transforms[view](
                probabilities[view * count + index : view * count + index + 1]
            )[0, 0].numpy()
            for view in range(len(transforms))
        ]
        predicted_iou = float(
            np.mean(
                [predicted_ious[view * count + index].item() for view in range(len(transforms))]
            )
        )
        x0, y0, x1, y1 = crop_bounds
        records.append(
            {
                "seed": seed,
                "detector_score": float(detector_score),
                "predicted_iou": predicted_iou,
                "bounds": crop_bounds,
                "probability": cv2.resize(
                    np.mean(views, axis=0),
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_LINEAR,
                ),
            }
        )
    return records


def evaluate_geometry_refiner_v4_oof(
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    indices: Mapping[str, Any],
    v3_checkpoint: Path,
    geometry_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare geometry on fold 0 while holding V3 selection fixed."""
    device = torch.device("cuda:0")
    v3_model = ProposalRefinerV3()
    v3_model.load_state_dict(
        torch.load(v3_checkpoint, map_location=device), strict=True
    )
    v3_model.to(device).eval()
    geometry_model = GeometryRefinerV4(v3_checkpoint)
    geometry_model.load_state_dict(
        torch.load(geometry_checkpoint, map_location=device), strict=True
    )
    geometry_model.to(device).eval()
    v3_cfg = replace(cfg, refiner_crop=448)
    frozen_selection = ProposalV3Setting(0.55, 0.30, 0.50)
    settings = geometry_v4_settings()
    grouped = catalog_by_filename(validation_samples)
    baseline_rows: list[dict[str, Any]] = []
    setting_rows: dict[str, list[dict[str, Any]]] = {
        setting.name: [] for setting in settings
    }
    for number, (filename, samples) in enumerate(sorted(grouped.items()), 1):
        image = cv2.imread(str(image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / filename)
        seeds = [mask_utils.decode(sample["seed_rle"]).astype(bool) for sample in samples]
        detector_scores = [float(sample["detector_score"]) for sample in samples]
        v3_records = proposal_v3_probability_records(
            image, seeds, detector_scores, v3_model, device, v3_cfg
        )
        geometry_records = geometry_v4_probability_records(
            image, seeds, detector_scores, geometry_model, device, cfg
        )
        if len(v3_records) != len(geometry_records):
            raise RuntimeError("V3 and V4 proposal counts differ")
        baseline_masks, _ = predictions_for_v3_setting(
            v3_records, frozen_selection, cfg.minimum_area
        )
        ground_truth = ground_truth_sets(filename, indices)
        for annotator, gt_masks in enumerate(ground_truth):
            baseline_rows.append(
                audit_metric_row(
                    filename,
                    0,
                    annotator,
                    "proposal_v3_frozen",
                    gt_masks,
                    baseline_masks,
                )
            )
        for setting in settings:
            blended_records = []
            for v3_record, geometry_record in zip(v3_records, geometry_records):
                if v3_record["bounds"] != geometry_record["bounds"]:
                    raise RuntimeError("V3 and V4 crop bounds differ")
                blended_records.append(
                    {
                        **v3_record,
                        "probability": (
                            (1.0 - setting.geometry_weight)
                            * v3_record["probability"]
                            + setting.geometry_weight
                            * geometry_record["probability"]
                        ),
                    }
                )
            mask_setting = replace(
                frozen_selection, mask_threshold=setting.mask_threshold
            )
            masks, _ = predictions_for_v3_setting(
                blended_records, mask_setting, cfg.minimum_area
            )
            for annotator, gt_masks in enumerate(ground_truth):
                setting_rows[setting.name].append(
                    audit_metric_row(
                        filename, 0, annotator, setting.name, gt_masks, masks
                    )
                )
        if number % 20 == 0 or number == len(grouped):
            print(
                {
                    "stage": "geometry_v4_oof",
                    "fold": 0,
                    "image": f"{number}/{len(grouped)}",
                },
                flush=True,
            )
    baseline = summary_with_sq(baseline_rows)
    summaries = {
        name: summary_with_sq(rows) for name, rows in setting_rows.items()
    }
    best_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["official_pq"],
            summaries[name]["sq"],
            -summaries[name]["mean_predictions"],
        ),
    )
    fold_dir = output_dir / "oof" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(
        fold_dir / "proposal_v3_frozen_by_annotator.csv", index=False
    )
    pd.DataFrame(setting_rows[best_name]).to_csv(
        fold_dir / "geometry_v4_best_by_annotator.csv", index=False
    )
    report = {
        "fold": 0,
        "selection": asdict(frozen_selection),
        "selection_frozen": True,
        "proposal_v3": baseline,
        "settings": summaries,
        "best_setting": best_name,
        "best": summaries[best_name],
        "official_pq_delta": summaries[best_name]["official_pq"]
        - baseline["official_pq"],
        "sq_delta": summaries[best_name]["sq"] - baseline["sq"],
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "geometry_v4_fold_complete", **report},
            separators=(",", ":"),
        ),
        flush=True,
    )
    del v3_model, geometry_model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_geometry_refiner_v4(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v3_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Fold-0 geometry gate only; never build test predictions or submit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(baseline_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.detector_conf = 0.05
    cfg.seed_nms_iou = 0.45
    cfg.refiner_crop = 512
    cfg.refiner_min_crop = 128
    cfg.refiner_batch = 3
    cfg.refiner_epochs = 4
    cfg.minimum_area = 96
    cfg.num_workers = 4
    v3_checkpoint = (
        v3_root / "models" / "fold_0" / "proposal_refiner_v3.pt"
    )
    smoke_check_geometry_refiner_v4(
        v3_checkpoint, paths["train"], indices, medoids, cfg
    )
    catalog = build_v3_oof_catalog(
        manifest, paths["train"], baseline_root, medoids, indices, cfg
    )
    train_files = set(manifest.loc[manifest["fold"] != 0, "filename"].astype(str))
    validation_files = set(
        manifest.loc[manifest["fold"] == 0, "filename"].astype(str)
    )
    train_samples = [
        value for value in catalog if value["filename"] in train_files
    ]
    validation_samples = [
        value for value in catalog if value["filename"] in validation_files
    ]
    catalog_report = {
        "samples": len(catalog),
        "train_positive": int(
            sum(float(value["existence"]) > 0 for value in train_samples)
        ),
        "validation_samples": len(validation_samples),
        "validation_positive": int(
            sum(float(value["existence"]) > 0 for value in validation_samples)
        ),
        "strictly_oof": True,
    }
    (output_dir / "catalog_report.json").write_text(
        json.dumps(catalog_report, indent=2), encoding="utf-8"
    )
    checkpoint, history = train_geometry_refiner_v4(
        train_samples,
        validation_samples,
        paths["train"],
        v3_checkpoint,
        cfg,
        output_dir,
    )
    report = evaluate_geometry_refiner_v4_oof(
        validation_samples,
        paths["train"],
        indices,
        v3_checkpoint,
        checkpoint,
        cfg,
        output_dir,
    )
    release_gate = {
        "official_pq": 0.450,
        "official_pq_delta": 0.006,
        "sq": 0.695,
        "maximum_runtime_hours": 4.5,
    }
    runtime_hours = (time.monotonic() - run_started) / 3600
    passed = (
        report["best"]["official_pq"] >= release_gate["official_pq"]
        and report["official_pq_delta"] >= release_gate["official_pq_delta"]
        and report["best"]["sq"] >= release_gate["sq"]
        and runtime_hours <= release_gate["maximum_runtime_hours"]
    )
    result = {
        "status": "geometry_gate_passed" if passed else "geometry_gate_rejected",
        "architecture": "v3_resnet18_highres",
        "initialization": str(v3_checkpoint),
        "crop": cfg.refiner_crop,
        "epochs": cfg.refiner_epochs,
        "release_gate": release_gate,
        "catalog": catalog_report,
        "best_crop_dice": float(history["validation_dice"].max()),
        "fold0": report,
        "runtime_hours": runtime_hours,
        "test_predictions_built": False,
        "submitted": False,
    }
    (output_dir / "geometry_v4_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"event": "geometry_v4_complete", **result}, separators=(",", ":")),
        flush=True,
    )
    return result


@dataclass(frozen=True)
class RecallV5Setting:
    max_extra: int
    existence_threshold: float
    score_threshold: float

    @property
    def name(self) -> str:
        return (
            f"k{self.max_extra}_e{self.existence_threshold:.2f}_"
            f"q{self.score_threshold:.2f}"
        )


def recall_v5_settings() -> tuple[RecallV5Setting, ...]:
    return tuple(
        RecallV5Setting(max_extra, existence, quality)
        for max_extra in (4, 8, 12)
        for existence in (0.40, 0.55, 0.70)
        for quality in (0.20, 0.30, 0.40)
    )


def build_recall_target_catalog(
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
) -> dict[str, list[dict[str, Any]]]:
    """Retain every annotation cluster, including one-annotator instances."""
    catalog: dict[str, list[dict[str, Any]]] = {}
    for filename in sorted(indices["image_ids_by_file"]):
        clusters = v3_consensus_clusters_for_file(
            filename, medoids, indices, cfg.consensus_match_iou
        )
        catalog[filename] = clusters
    return catalog


def write_recall_detector_split(
    output: Path,
    filenames: Sequence[str],
    image_dir: Path,
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    images_dir, labels_dir = output / "images", output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        destination = images_dir / filename
        if not destination.exists():
            os.symlink(image_dir / filename, destination)
        labels = []
        for cluster in catalog[filename]:
            polygon = mask_to_polygon(cluster["representative"])
            if polygon is not None:
                labels.append(polygon)
        (labels_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(labels), encoding="utf-8"
        )


def prepare_recall_detector_fold_v5(
    manifest: pd.DataFrame,
    image_dir: Path,
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> Path:
    fold_dir = output_dir / "detector_data" / "fold_0"
    train_files = manifest.loc[manifest["fold"] != 0, "filename"].astype(str).tolist()
    validation_files = manifest.loc[
        manifest["fold"] == 0, "filename"
    ].astype(str).tolist()
    write_recall_detector_split(
        fold_dir / "train", train_files, image_dir, catalog
    )
    write_recall_detector_split(
        fold_dir / "val", validation_files, image_dir, catalog
    )
    yaml_path = fold_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {fold_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    return yaml_path


def train_recall_detector_v5(
    yaml_path: Path, cfg: CFG, output_dir: Path
) -> Path:
    from ultralytics import YOLO

    model_dir = output_dir / "models" / "fold_0"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "recall_detector.pt"
    model = YOLO(cfg.detector_model)
    gpu_count = torch.cuda.device_count()
    device: int | str = (
        ",".join(map(str, range(gpu_count)))
        if gpu_count > 1
        else (0 if gpu_count else "cpu")
    )
    print(
        {
            "stage": "recall_v5_detector_train",
            "fold": 0,
            "gpus": gpu_count,
            "device": device,
            "epochs": cfg.detector_epochs,
            "image_size": cfg.detector_size,
            "batch": cfg.detector_batch,
        },
        flush=True,
    )
    model.train(
        data=str(yaml_path),
        epochs=cfg.detector_epochs,
        imgsz=cfg.detector_size,
        batch=cfg.detector_batch,
        device=device,
        patience=cfg.detector_patience,
        seed=cfg.seed + 500,
        single_cls=True,
        mask_ratio=cfg.detector_mask_ratio,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.30,
        degrees=18,
        fliplr=0.5,
        flipud=0.5,
        scale=0.35,
        translate=0.10,
        mosaic=0.35,
        close_mosaic=4,
        workers=2,
        project=str(output_dir / "detector_runs"),
        name="fold_0_recall",
        exist_ok=True,
        verbose=True,
    )
    shutil.copy2(str(model.trainer.best), checkpoint)
    print(
        {"stage": "recall_v5_detector_ready", "checkpoint": str(checkpoint)},
        flush=True,
    )
    return checkpoint


def discover_recall_detector_v5_checkpoint() -> Path | None:
    """Find the immutable checkpoint exported by the completed V5 training run."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    candidates = sorted(input_root.rglob("recall_detector.pt"))
    preferred = [
        path
        for path in candidates
        if "filament-recall-detector-v5-fold0" in str(path)
    ]
    return (preferred or candidates)[0] if candidates else None


@torch.no_grad()
def build_v3_validation_catalog_v5(
    manifest: pd.DataFrame,
    image_dir: Path,
    baseline_root: Path,
    medoids: Mapping[str, str],
    indices: Mapping[str, Any],
    cfg: CFG,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    device = torch.device("cuda:0")
    detector = YOLO(str(baseline_root / "models" / "fold_0" / "detector.pt"))
    filenames = manifest.loc[manifest["fold"] == 0, "filename"].astype(str).tolist()
    samples: list[dict[str, Any]] = []
    for number, filename in enumerate(filenames, 1):
        samples.extend(
            build_v3_samples_for_file(
                filename,
                image_dir,
                detector,
                device,
                medoids,
                indices,
                cfg,
                0,
            )
        )
        if number % 25 == 0 or number == len(filenames):
            print(
                {
                    "stage": "recall_v5_baseline_catalog",
                    "image": f"{number}/{len(filenames)}",
                    "samples": len(samples),
                },
                flush=True,
            )
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return samples


def novel_recall_candidates_v5(
    baseline_seeds: Sequence[np.ndarray],
    recall_seeds: Sequence[np.ndarray],
    recall_scores: Sequence[float],
    minimum_area: int,
    maximum: int = 12,
) -> tuple[list[np.ndarray], list[float]]:
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []
    for index in sorted(
        range(len(recall_seeds)),
        key=lambda value: float(recall_scores[value]),
        reverse=True,
    ):
        seed = recall_seeds[index].astype(bool)
        if int(seed.sum()) < minimum_area:
            continue
        if any(mask_iou(seed, base) >= 0.45 for base in baseline_seeds):
            continue
        if any(mask_iou(seed, other) >= 0.45 for other in kept_masks):
            continue
        kept_masks.append(seed)
        kept_scores.append(max(float(recall_scores[index]), 0.05))
        if len(kept_masks) >= maximum:
            break
    return kept_masks, kept_scores


def update_proposal_recall_v5(
    totals: dict[str, float],
    ground_truth: Sequence[np.ndarray],
    seeds: Sequence[np.ndarray],
) -> None:
    for truth in ground_truth:
        best = max((mask_iou(truth, seed) for seed in seeds), default=0.0)
        totals["instances"] += 1
        totals["iou_sum"] += best
        totals["recall_010"] += float(best >= 0.10)
        totals["recall_050"] += float(best >= 0.50)


def finalize_proposal_recall_v5(totals: Mapping[str, float]) -> dict[str, float]:
    count = max(float(totals["instances"]), 1.0)
    return {
        "instances": int(totals["instances"]),
        "mean_best_iou": float(totals["iou_sum"]) / count,
        "recall_at_010": float(totals["recall_010"]) / count,
        "recall_at_050": float(totals["recall_050"]) / count,
    }


def evaluate_recall_detector_v5(
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    indices: Mapping[str, Any],
    v3_checkpoint: Path,
    recall_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> dict[str, Any]:
    from ultralytics import YOLO

    v3_device = torch.device("cuda:0")
    recall_device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    )
    v3_model = ProposalRefinerV3()
    v3_model.load_state_dict(
        torch.load(v3_checkpoint, map_location=v3_device), strict=True
    )
    v3_model.to(v3_device).eval()
    recall_detector = YOLO(str(recall_checkpoint))
    print(
        {
            "stage": "recall_v5_evaluation_devices",
            "proposal_refiner": str(v3_device),
            "recall_detector": str(recall_device),
        },
        flush=True,
    )
    grouped = catalog_by_filename(validation_samples)
    baseline_setting = ProposalV3Setting(0.55, 0.30, 0.50)
    settings = recall_v5_settings()
    baseline_rows: list[dict[str, Any]] = []
    setting_rows: dict[str, list[dict[str, Any]]] = {
        setting.name: [] for setting in settings
    }
    proposal_totals = {
        "baseline": {key: 0.0 for key in ("instances", "iou_sum", "recall_010", "recall_050")},
        "combined": {key: 0.0 for key in ("instances", "iou_sum", "recall_010", "recall_050")},
    }
    candidate_diagnostics = []
    for number, (filename, samples) in enumerate(sorted(grouped.items()), 1):
        image_path = image_dir / filename
        baseline_seeds = [
            mask_utils.decode(sample["seed_rle"]).astype(bool) for sample in samples
        ]
        baseline_scores = [float(sample["detector_score"]) for sample in samples]
        image, recall_seeds, recall_scores = raw_legacy_candidates(
            image_path, recall_detector, recall_device, cfg, confidence=0.01
        )
        extra_seeds, extra_scores = novel_recall_candidates_v5(
            baseline_seeds,
            recall_seeds,
            recall_scores,
            cfg.minimum_area,
            maximum=12,
        )
        all_seeds = baseline_seeds + extra_seeds
        all_scores = baseline_scores + extra_scores
        records = proposal_v3_probability_records(
            image, all_seeds, all_scores, v3_model, v3_device, cfg
        )
        baseline_records = records[: len(baseline_seeds)]
        baseline_masks, _ = predictions_for_v3_setting(
            baseline_records, baseline_setting, cfg.minimum_area
        )
        ground_truth_sets_for_file = ground_truth_sets(filename, indices)
        for annotator, ground_truth in enumerate(ground_truth_sets_for_file):
            baseline_rows.append(
                audit_metric_row(
                    filename,
                    0,
                    annotator,
                    "proposal_v3_frozen",
                    ground_truth,
                    baseline_masks,
                )
            )
            update_proposal_recall_v5(
                proposal_totals["baseline"], ground_truth, baseline_seeds
            )
            update_proposal_recall_v5(
                proposal_totals["combined"], ground_truth, all_seeds
            )
        for setting in settings:
            selected_records = records[
                : len(baseline_seeds) + min(setting.max_extra, len(extra_seeds))
            ]
            v3_setting = ProposalV3Setting(
                setting.existence_threshold,
                setting.score_threshold,
                0.50,
            )
            masks, _ = predictions_for_v3_setting(
                selected_records, v3_setting, cfg.minimum_area
            )
            for annotator, ground_truth in enumerate(ground_truth_sets_for_file):
                setting_rows[setting.name].append(
                    audit_metric_row(
                        filename,
                        0,
                        annotator,
                        setting.name,
                        ground_truth,
                        masks,
                    )
                )
        candidate_diagnostics.append(
            {
                "filename": filename,
                "baseline_candidates": len(baseline_seeds),
                "recall_raw_candidates": len(recall_seeds),
                "novel_candidates": len(extra_seeds),
            }
        )
        if number % 20 == 0 or number == len(grouped):
            print(
                {
                    "stage": "recall_v5_oof",
                    "image": f"{number}/{len(grouped)}",
                    "mean_novel": round(
                        float(np.mean([row["novel_candidates"] for row in candidate_diagnostics])),
                        2,
                    ),
                },
                flush=True,
            )
    baseline = summary_with_sq(baseline_rows)
    summaries = {
        name: summary_with_sq(rows) for name, rows in setting_rows.items()
    }
    best_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["official_pq"],
            summaries[name]["rq"],
            summaries[name]["sq"],
        ),
    )
    oof_dir = output_dir / "oof" / "fold_0"
    oof_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(
        oof_dir / "proposal_v3_frozen_by_annotator.csv", index=False
    )
    pd.DataFrame(setting_rows[best_name]).to_csv(
        oof_dir / "recall_v5_best_by_annotator.csv", index=False
    )
    pd.DataFrame(candidate_diagnostics).to_csv(
        oof_dir / "candidate_diagnostics.csv", index=False
    )
    report = {
        "fold": 0,
        "proposal_v3": baseline,
        "settings": summaries,
        "best_setting": best_name,
        "best": summaries[best_name],
        "official_pq_delta": summaries[best_name]["official_pq"]
        - baseline["official_pq"],
        "tp_delta": summaries[best_name]["tp"] - baseline["tp"],
        "fp_delta": summaries[best_name]["fp"] - baseline["fp"],
        "fn_delta": summaries[best_name]["fn"] - baseline["fn"],
        "proposal_recall": {
            name: finalize_proposal_recall_v5(values)
            for name, values in proposal_totals.items()
        },
        "mean_novel_candidates": float(
            np.mean([row["novel_candidates"] for row in candidate_diagnostics])
        ),
    }
    (oof_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"event": "recall_v5_fold_complete", **report}, separators=(",", ":")),
        flush=True,
    )
    del v3_model, recall_detector
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_recall_detector_v5(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v3_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Fold-0 high-recall detector gate; never infer test or submit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(baseline_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.detector_model = "/kaggle/working/yolo11s-seg.pt"
    cfg.detector_size = 1024
    cfg.detector_epochs = 18
    cfg.detector_batch = 2
    cfg.detector_patience = 5
    cfg.detector_mask_ratio = 4
    cfg.refiner_crop = 448
    cfg.refiner_min_crop = 128
    cfg.minimum_area = 96
    recall_catalog = build_recall_target_catalog(medoids, indices, cfg)
    production_catalog = build_consensus_catalog(indices, medoids, cfg)
    target_report = {
        "physical_images": len(recall_catalog),
        "production_instances": int(sum(map(len, production_catalog.values()))),
        "recall_instances": int(sum(map(len, recall_catalog.values()))),
        "added_instances": int(
            sum(map(len, recall_catalog.values()))
            - sum(map(len, production_catalog.values()))
        ),
        "single_annotator_instances": int(
            sum(
                len(cluster["target_rles"]) == 1
                for clusters in recall_catalog.values()
                for cluster in clusters
            )
        ),
        "strict_fold0_holdout": True,
    }
    (output_dir / "target_report.json").write_text(
        json.dumps(target_report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"event": "recall_v5_targets", **target_report}, separators=(",", ":")),
        flush=True,
    )
    recall_checkpoint = discover_recall_detector_v5_checkpoint()
    if recall_checkpoint is None:
        raise FileNotFoundError(
            "Missing recall_detector.pt input; attach the private "
            "filament-recall-detector-v5-fold0 dataset"
        )
    print(
        {
            "stage": "recall_v5_checkpoint_reused",
            "checkpoint": str(recall_checkpoint),
            "training_skipped": True,
        },
        flush=True,
    )
    validation_samples = build_v3_validation_catalog_v5(
        manifest,
        paths["train"],
        baseline_root,
        medoids,
        indices,
        cfg,
    )
    v3_checkpoint = (
        v3_root / "models" / "fold_0" / "proposal_refiner_v3.pt"
    )
    report = evaluate_recall_detector_v5(
        validation_samples,
        paths["train"],
        indices,
        v3_checkpoint,
        recall_checkpoint,
        cfg,
        output_dir,
    )
    runtime_hours = (time.monotonic() - run_started) / 3600
    release_gate = {
        "official_pq": 0.452,
        "official_pq_delta": 0.007,
        "minimum_tp_delta": 20,
        "maximum_fp_delta": 15,
        "maximum_runtime_hours": 3.5,
    }
    passed = (
        report["best"]["official_pq"] >= release_gate["official_pq"]
        and report["official_pq_delta"] >= release_gate["official_pq_delta"]
        and report["tp_delta"] >= release_gate["minimum_tp_delta"]
        and report["fp_delta"] <= release_gate["maximum_fp_delta"]
        and runtime_hours <= release_gate["maximum_runtime_hours"]
    )
    result = {
        "status": "recall_gate_passed" if passed else "recall_gate_rejected",
        "architecture": "yolo11s-seg_all_annotation_clusters_plus_frozen_v3",
        "checkpoint_reused": str(recall_checkpoint),
        "training_skipped": True,
        "release_gate": release_gate,
        "targets": target_report,
        "fold0": report,
        "runtime_hours": runtime_hours,
        "test_predictions_built": False,
        "submitted": False,
    }
    (output_dir / "recall_v5_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"event": "recall_v5_complete", **result}, separators=(",", ":")),
        flush=True,
    )
    return result


NOVEL_SELECTOR_V6_FEATURES = (
    "detector_score",
    "log_area",
    "fill",
    "aspect",
    "radius",
    "contrast",
    "v3_existence",
    "v3_predicted_iou",
    "v3_expected_quality",
    "foreground_probability",
    "output_to_seed_area",
    "max_seed_baseline_iou",
    "max_output_baseline_iou",
    "is_refined",
)


def v3_record_mask_v6(
    record: Mapping[str, Any], minimum_area: int, threshold: float = 0.50
) -> np.ndarray | None:
    crop_mask = np.asarray(record["probability"]) >= threshold
    if int(crop_mask.sum()) < minimum_area:
        return None
    full_mask = np.zeros_like(record["seed"], dtype=bool)
    x0, y0, x1, y1 = record["bounds"]
    full_mask[y0:y1, x0:x1] = crop_mask
    return full_mask


def novel_selector_features_v6(
    image: np.ndarray,
    seed: np.ndarray,
    output_mask: np.ndarray,
    record: Mapping[str, Any],
    baseline_seeds: Sequence[np.ndarray],
    baseline_masks: Sequence[np.ndarray],
    is_refined: bool,
) -> list[float]:
    metadata = proposal_v3_metadata(
        image, seed, float(record["detector_score"])
    )
    crop_mask = np.asarray(record["probability"]) >= 0.50
    foreground_probability = (
        float(np.asarray(record["probability"])[crop_mask].mean())
        if crop_mask.any()
        else 0.0
    )
    seed_area = max(int(seed.sum()), 1)
    existence = float(record["existence"])
    predicted_iou = float(record["predicted_iou"])
    return [
        *[float(value) for value in metadata],
        existence,
        predicted_iou,
        existence * predicted_iou,
        foreground_probability,
        float(output_mask.sum() / seed_area),
        max((mask_iou(seed, mask) for mask in baseline_seeds), default=0.0),
        max((mask_iou(output_mask, mask) for mask in baseline_masks), default=0.0),
        float(is_refined),
    ]


def novel_candidate_help_rate_v6(
    candidate: np.ndarray,
    baseline_masks: Sequence[np.ndarray],
    ground_truth_versions: Sequence[Sequence[np.ndarray]],
) -> float:
    helpful = 0
    for ground_truth in ground_truth_versions:
        candidate_overlaps = pairwise_iou(ground_truth, [candidate])[:, 0]
        if not candidate_overlaps.size:
            continue
        target_index = int(candidate_overlaps.argmax())
        if float(candidate_overlaps[target_index]) <= 0.50:
            continue
        already_covered = max(
            (
                mask_iou(ground_truth[target_index], baseline)
                for baseline in baseline_masks
            ),
            default=0.0,
        )
        helpful += int(already_covered <= 0.50)
    return helpful / max(len(ground_truth_versions), 1)


def crossfit_novel_selector_v6(
    candidates: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray([row["features"] for row in candidates], dtype=np.float32)
    labels = np.asarray([row["label"] for row in candidates], dtype=np.int64)
    groups = np.asarray([row["filename"] for row in candidates])
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Novel-selector candidates do not contain both classes")
    group_count = len(np.unique(groups))
    split_count = min(5, group_count)
    if split_count < 2:
        raise RuntimeError("Novel-selector cross-fitting needs at least two images")
    probabilities = np.zeros(len(candidates), dtype=np.float32)
    splitter = GroupKFold(n_splits=split_count)
    for split, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups), 1
    ):
        if len(np.unique(labels[train_indices])) != 2:
            raise RuntimeError(f"Novel-selector split {split} has only one class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed + split,
            ),
        )
        model.fit(features[train_indices], labels[train_indices])
        probabilities[validation_indices] = model.predict_proba(
            features[validation_indices]
        )[:, 1]
    for row, probability in zip(candidates, probabilities):
        row["selector_probability"] = float(probability)
    return {
        "candidates": len(candidates),
        "positive_candidates": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "group_folds": split_count,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "probability_quantiles": {
            f"q{int(quantile * 100):02d}": float(np.quantile(probabilities, quantile))
            for quantile in (0.50, 0.75, 0.90, 0.95, 0.99)
        },
    }


def append_novel_masks_v6(
    baseline_masks: Sequence[np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
    probability_threshold: float,
    maximum: int,
    minimum_area: int,
) -> tuple[list[np.ndarray], list[Mapping[str, Any]]]:
    by_proposal: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        if float(candidate["selector_probability"]) < probability_threshold:
            continue
        proposal = int(candidate["proposal"])
        previous = by_proposal.get(proposal)
        if previous is None or float(candidate["selector_probability"]) > float(
            previous["selector_probability"]
        ):
            by_proposal[proposal] = candidate
    ranked = sorted(
        by_proposal.values(),
        key=lambda row: float(row["selector_probability"]),
        reverse=True,
    )[:maximum]
    output = [mask.copy() for mask in baseline_masks]
    assigned = np.logical_or.reduce(output) if output else None
    accepted: list[Mapping[str, Any]] = []
    for candidate in ranked:
        mask = mask_utils.decode(candidate["mask_rle"]).astype(bool)
        clean = mask if assigned is None else mask & ~assigned
        if int(clean.sum()) < minimum_area:
            continue
        output.append(clean)
        accepted.append(candidate)
        assigned = clean.copy() if assigned is None else assigned | clean
    return output, accepted


@torch.no_grad()
def evaluate_novel_selector_v6(
    validation_samples: Sequence[Mapping[str, Any]],
    image_dir: Path,
    indices: Mapping[str, Any],
    v3_checkpoint: Path,
    recall_checkpoint: Path,
    cfg: CFG,
    output_dir: Path,
) -> dict[str, Any]:
    from ultralytics import YOLO

    v3_device = torch.device("cuda:0")
    recall_device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    )
    v3_model = ProposalRefinerV3()
    v3_model.load_state_dict(
        torch.load(v3_checkpoint, map_location=v3_device), strict=True
    )
    v3_model.to(v3_device).eval()
    recall_detector = YOLO(str(recall_checkpoint))
    grouped = catalog_by_filename(validation_samples)
    baseline_setting = ProposalV3Setting(0.55, 0.30, 0.50)
    image_records: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for number, (filename, samples) in enumerate(sorted(grouped.items()), 1):
        image_path = image_dir / filename
        baseline_seeds = [
            mask_utils.decode(sample["seed_rle"]).astype(bool) for sample in samples
        ]
        baseline_scores = [float(sample["detector_score"]) for sample in samples]
        image, recall_seeds, recall_scores = raw_legacy_candidates(
            image_path, recall_detector, recall_device, cfg, confidence=0.01
        )
        extra_seeds, extra_scores = novel_recall_candidates_v5(
            baseline_seeds,
            recall_seeds,
            recall_scores,
            cfg.minimum_area,
            maximum=12,
        )
        records = proposal_v3_probability_records(
            image,
            baseline_seeds + extra_seeds,
            baseline_scores + extra_scores,
            v3_model,
            v3_device,
            cfg,
        )
        baseline_records = records[: len(baseline_seeds)]
        baseline_masks, _ = predictions_for_v3_setting(
            baseline_records, baseline_setting, cfg.minimum_area
        )
        baseline_union = (
            np.logical_or.reduce(baseline_masks) if baseline_masks else None
        )
        ground_truth_versions = ground_truth_sets(filename, indices)
        for annotator, ground_truth in enumerate(ground_truth_versions):
            baseline_rows.append(
                audit_metric_row(
                    filename,
                    0,
                    annotator,
                    "proposal_v3_frozen",
                    ground_truth,
                    baseline_masks,
                )
            )
        per_image_candidates: list[dict[str, Any]] = []
        for proposal, (seed, record) in enumerate(
            zip(extra_seeds, records[len(baseline_seeds) :])
        ):
            refined = v3_record_mask_v6(record, cfg.minimum_area)
            variants = [("seed", seed, False)]
            if refined is not None:
                variants.append(("refined", refined, True))
            for variant, output_mask, is_refined in variants:
                clean_output = (
                    output_mask
                    if baseline_union is None
                    else output_mask & ~baseline_union
                )
                if int(clean_output.sum()) < cfg.minimum_area:
                    continue
                help_rate = novel_candidate_help_rate_v6(
                    clean_output, baseline_masks, ground_truth_versions
                )
                candidate = {
                    "filename": filename,
                    "proposal": proposal,
                    "variant": variant,
                    "mask_rle": binary_rle(clean_output),
                    "features": novel_selector_features_v6(
                        image,
                        seed,
                        output_mask,
                        record,
                        baseline_seeds,
                        baseline_masks,
                        is_refined,
                    ),
                    "help_rate": help_rate,
                    "label": int(help_rate >= 0.50),
                }
                candidates.append(candidate)
                per_image_candidates.append(candidate)
        image_records[filename] = {
            "baseline_rles": [binary_rle(mask) for mask in baseline_masks],
            "candidates": per_image_candidates,
        }
        if number % 20 == 0 or number == len(grouped):
            print(
                {
                    "stage": "novel_selector_v6_catalog",
                    "image": f"{number}/{len(grouped)}",
                    "candidate_variants": len(candidates),
                },
                flush=True,
            )
    selector_report = crossfit_novel_selector_v6(candidates, cfg.seed + 600)
    probabilities = np.asarray(
        [float(row["selector_probability"]) for row in candidates]
    )
    thresholds = sorted(
        {
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            1.01,
            *[
                round(float(np.quantile(probabilities, quantile)), 4)
                for quantile in (0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
            ],
        }
    )
    setting_rows: dict[str, list[dict[str, Any]]] = {}
    selection_reports: dict[str, dict[str, Any]] = {}
    for maximum in (1, 2, 4):
        for threshold in thresholds:
            name = f"k{maximum}_p{threshold:.4f}"
            rows: list[dict[str, Any]] = []
            accepted_rows: list[Mapping[str, Any]] = []
            for filename, image_record in sorted(image_records.items()):
                baseline_masks = [
                    mask_utils.decode(rle).astype(bool)
                    for rle in image_record["baseline_rles"]
                ]
                masks, accepted = append_novel_masks_v6(
                    baseline_masks,
                    image_record["candidates"],
                    threshold,
                    maximum,
                    cfg.minimum_area,
                )
                accepted_rows.extend(accepted)
                for annotator, ground_truth in enumerate(
                    ground_truth_sets(filename, indices)
                ):
                    rows.append(
                        audit_metric_row(
                            filename,
                            0,
                            annotator,
                            name,
                            ground_truth,
                            masks,
                        )
                    )
            setting_rows[name] = rows
            selection_reports[name] = {
                "threshold": threshold,
                "maximum_per_image": maximum,
                "accepted": len(accepted_rows),
                "label_precision": float(
                    np.mean([row["label"] for row in accepted_rows])
                )
                if accepted_rows
                else 1.0,
                "mean_help_rate": float(
                    np.mean([row["help_rate"] for row in accepted_rows])
                )
                if accepted_rows
                else 0.0,
            }
    baseline = summary_with_sq(baseline_rows)
    summaries = {
        name: summary_with_sq(rows) for name, rows in setting_rows.items()
    }
    best_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["official_pq"],
            summaries[name]["rq"],
            summaries[name]["sq"],
        ),
    )
    baseline_control_names = [name for name in summaries if name.endswith("p1.0100")]
    for name in baseline_control_names:
        if not np.isclose(
            summaries[name]["official_pq"], baseline["official_pq"], atol=1e-12
        ):
            raise AssertionError("V6 no-addition control does not reproduce V3")
    candidate_frame_rows = []
    for row in candidates:
        output = {
            key: row[key]
            for key in (
                "filename",
                "proposal",
                "variant",
                "help_rate",
                "label",
                "selector_probability",
            )
        }
        output.update(dict(zip(NOVEL_SELECTOR_V6_FEATURES, row["features"])))
        candidate_frame_rows.append(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_frame_rows).to_csv(
        output_dir / "novel_candidates_crossfit.csv", index=False
    )
    pd.DataFrame(baseline_rows).to_csv(
        output_dir / "proposal_v3_frozen_by_annotator.csv", index=False
    )
    pd.DataFrame(setting_rows[best_name]).to_csv(
        output_dir / "novel_selector_v6_best_by_annotator.csv", index=False
    )
    report = {
        "fold": 0,
        "evaluation": "grouped_5fold_crossfit_selector_on_strict_oof_base_predictions",
        "proposal_v3": baseline,
        "selector": selector_report,
        "best_setting": best_name,
        "best_selection": selection_reports[best_name],
        "best": summaries[best_name],
        "official_pq_delta": summaries[best_name]["official_pq"]
        - baseline["official_pq"],
        "tp_delta": summaries[best_name]["tp"] - baseline["tp"],
        "fp_delta": summaries[best_name]["fp"] - baseline["fp"],
        "fn_delta": summaries[best_name]["fn"] - baseline["fn"],
        "settings": summaries,
        "selection_reports": selection_reports,
        "baseline_control_passed": True,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "novel_selector_v6_complete", **report},
            separators=(",", ":"),
        ),
        flush=True,
    )
    del v3_model, recall_detector
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_novel_selector_v6(
    paths: Mapping[str, Path],
    baseline_root: Path,
    v3_root: Path,
    output_dir: Path,
    run_started: float,
) -> dict[str, Any]:
    """Cross-fitted selector for additive V5 proposals; never infer test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths["json"].open("r", encoding="utf-8") as handle:
        indices = build_indices(json.load(handle))
    medoids = choose_medoid_records(indices)
    manifest = pd.read_csv(baseline_root / "fold_manifest.csv")
    cfg = legacy_inference_cfg()
    cfg.detector_size = 1024
    cfg.refiner_crop = 448
    cfg.refiner_min_crop = 128
    cfg.minimum_area = 96
    recall_checkpoint = discover_recall_detector_v5_checkpoint()
    if recall_checkpoint is None:
        raise FileNotFoundError(
            "Missing recall_detector.pt input; attach the private "
            "filament-recall-detector-v5-fold0 dataset"
        )
    validation_samples = build_v3_validation_catalog_v5(
        manifest,
        paths["train"],
        baseline_root,
        medoids,
        indices,
        cfg,
    )
    report = evaluate_novel_selector_v6(
        validation_samples,
        paths["train"],
        indices,
        v3_root / "models" / "fold_0" / "proposal_refiner_v3.pt",
        recall_checkpoint,
        cfg,
        output_dir,
    )
    runtime_hours = (time.monotonic() - run_started) / 3600
    release_gate = {
        "official_pq": 0.452,
        "official_pq_delta": 0.007,
        "minimum_tp_delta": 12,
        "maximum_fp_delta": 15,
        "maximum_runtime_hours": 3.5,
    }
    passed = (
        report["best"]["official_pq"] >= release_gate["official_pq"]
        and report["official_pq_delta"] >= release_gate["official_pq_delta"]
        and report["tp_delta"] >= release_gate["minimum_tp_delta"]
        and report["fp_delta"] <= release_gate["maximum_fp_delta"]
        and runtime_hours <= release_gate["maximum_runtime_hours"]
    )
    result = {
        "status": "selector_gate_passed" if passed else "selector_gate_rejected",
        "architecture": "frozen_v3_plus_crossfitted_additive_novel_selector",
        "checkpoint_reused": str(recall_checkpoint),
        "release_gate": release_gate,
        "fold0": report,
        "runtime_hours": runtime_hours,
        "test_predictions_built": False,
        "submitted": False,
    }
    (output_dir / "novel_selector_v6_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "novel_selector_v6_gate", **result},
            separators=(",", ":"),
        ),
        flush=True,
    )
    return result


def validate_metric() -> None:
    first = np.zeros((16, 16), bool)
    first[1:5, 1:5] = True
    second = np.zeros((16, 16), bool)
    second[8:14, 8:14] = True
    assert official_pq([first, second], [second, first])["pq"] == 1.0
    assert np.isclose(official_pq([first, second], [first])["pq"], 2 / 3)
    assert np.isclose(official_pq([first], [first, second])["pq"], 2 / 3)
    half = np.zeros((16, 16), bool)
    half[1:5, 3:7] = True
    assert official_pq([first], [half])["pq"] == 0.0
    assert official_pq([first], [])["pq"] == 0.0


def main() -> None:
    run_started = time.monotonic()
    cfg = CFG()
    seed_everything(cfg.seed)
    print({"config": asdict(cfg), "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count()})
    validate_metric()
    if not KAGGLE_DATA_AVAILABLE:
        print("Kaggle input unavailable; metric self-tests passed.")
        return
    if cfg.require_dual_gpu and not cfg.debug and torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"This full run requires exactly two CUDA devices, found {torch.cuda.device_count()}"
        )
    paths = discover_paths()
    checkpoint_roots = discover_parent_checkpoint_roots()
    domain_backbone = discover_domain_backbone()
    if RUN_MODE == "proposal-v23-hybrid-sweep":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "V2/V3 hybrid sweep requires the five baseline detector folds"
            )
        proposal_v2_root = discover_proposal_v2_root()
        proposal_v3_root = discover_proposal_v3_full_root()
        if proposal_v2_root is None:
            raise FileNotFoundError(
                "V2/V3 hybrid sweep requires the promoted five-fold V2 kernel"
            )
        if proposal_v3_root is None:
            raise FileNotFoundError(
                "V2/V3 hybrid sweep requires the completed five-fold V3 kernel"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v2_root": str(proposal_v2_root),
                "proposal_v3_root": str(proposal_v3_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "candidates": [
                    "hybrid012_s3_medoid",
                    "hybrid012_s3_majority",
                    "v3all_s3_medoid",
                    "v3all_s3_majority",
                ],
                "automatic_submission": False,
            }
        )
        report = run_proposal_v23_hybrid_sweep(
            paths,
            checkpoint_roots["baseline"],
            proposal_v2_root,
            proposal_v3_root,
            Path("/kaggle/working/proposal_v23_hybrid_sweep"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if RUN_MODE == "proposal-refiner-v3-resume":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Proposal V3 resume requires the five baseline detector folds"
            )
        proposal_v2_root = discover_proposal_v2_root()
        proposal_v3_root = discover_proposal_v3_root()
        if proposal_v2_root is None:
            raise FileNotFoundError(
                "Proposal V3 resume requires the promoted five-fold V2 kernel"
            )
        if proposal_v3_root is None:
            raise FileNotFoundError(
                "Proposal V3 resume requires the proven fold-0 V3 kernel"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v2_root": str(proposal_v2_root),
                "proposal_v3_resume_root": str(proposal_v3_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "reused_folds": [0],
                "trained_folds": [1, 2, 3, 4],
                "automatic_submission": False,
            }
        )
        report = run_proposal_refiner_v3(
            paths,
            checkpoint_roots["baseline"],
            proposal_v2_root,
            Path("/kaggle/working/proposal_refiner_v3_full"),
            run_started,
            resume_root=proposal_v3_root,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if RUN_MODE == "proposal-v2-policy-sweep":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Proposal v2 policy sweep requires the five baseline folds"
            )
        proposal_v2_root = discover_proposal_v2_root()
        if proposal_v2_root is None:
            raise FileNotFoundError(
                "Proposal v2 policy sweep requires the promoted five-fold V2 kernel"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v2_root": str(proposal_v2_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "policies": [
                    "s2_medoid",
                    "s3_majority",
                    "s4_medoid",
                    "s5_medoid",
                ],
                "automatic_submission": False,
            }
        )
        report = run_proposal_v2_policy_sweep(
            paths,
            checkpoint_roots["baseline"],
            proposal_v2_root,
            Path("/kaggle/working/proposal_v2_policy_sweep"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round(
                    (time.monotonic() - run_started) / 3600, 3
                ),
            }
        )
        return
    if RUN_MODE == "novel-selector-v6":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Novel selector v6 requires the five baseline detector folds"
            )
        proposal_v3_root = discover_proposal_v3_root()
        if proposal_v3_root is None:
            raise FileNotFoundError(
                "Novel selector v6 requires the promoted fold-0 v3 checkpoint"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v3_root": str(proposal_v3_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "baseline_predictions_frozen": True,
                "selector_crossfitted": True,
                "automatic_submission": False,
            }
        )
        report = run_novel_selector_v6(
            paths,
            checkpoint_roots["baseline"],
            proposal_v3_root,
            Path("/kaggle/working/novel_selector_v6"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round(
                    (time.monotonic() - run_started) / 3600, 3
                ),
            }
        )
        return
    if RUN_MODE == "recall-detector-v5":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Recall detector v5 requires the five baseline detector folds"
            )
        proposal_v3_root = discover_proposal_v3_root()
        if proposal_v3_root is None:
            raise FileNotFoundError(
                "Recall detector v5 requires the promoted fold-0 v3 checkpoint"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v3_root": str(proposal_v3_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "automatic_submission": False,
            }
        )
        report = run_recall_detector_v5(
            paths,
            checkpoint_roots["baseline"],
            proposal_v3_root,
            Path("/kaggle/working/recall_detector_v5"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if RUN_MODE == "geometry-refiner-v4":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Geometry refiner v4 requires the five baseline detector folds"
            )
        proposal_v3_root = discover_proposal_v3_root()
        if proposal_v3_root is None:
            raise FileNotFoundError(
                "Geometry refiner v4 requires the promoted fold-0 v3 checkpoint"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v3_root": str(proposal_v3_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "selection_frozen": True,
                "automatic_submission": False,
            }
        )
        report = run_geometry_refiner_v4(
            paths,
            checkpoint_roots["baseline"],
            proposal_v3_root,
            Path("/kaggle/working/geometry_refiner_v4"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if RUN_MODE == "proposal-refiner-v3":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Proposal refiner v3 requires the five baseline detector folds"
            )
        proposal_v2_root = discover_proposal_v2_root()
        if proposal_v2_root is None:
            raise FileNotFoundError(
                "Proposal refiner v3 requires the five promoted v2 refiner folds"
            )
        print(
            {
                "mode": RUN_MODE,
                "baseline_root": str(checkpoint_roots["baseline"]),
                "proposal_v2_root": str(proposal_v2_root),
                "cuda_devices": list(range(torch.cuda.device_count())),
                "automatic_submission": False,
            }
        )
        report = run_proposal_refiner_v3(
            paths,
            checkpoint_roots["baseline"],
            proposal_v2_root,
            Path("/kaggle/working/proposal_refiner_v3"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "submitted": False,
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if RUN_MODE == "proposal-refiner-v2":
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "Proposal refiner v2 requires the five baseline detector/refiner folds"
            )
        print(
            {
                "mode": RUN_MODE,
                "checkpoint_root": str(checkpoint_roots["baseline"]),
                "cuda_devices": list(range(torch.cuda.device_count())),
            }
        )
        report = run_proposal_refiner_v2(
            paths,
            checkpoint_roots["baseline"],
            Path("/kaggle/working/proposal_refiner_v2"),
            run_started,
        )
        print(
            {
                "mode": RUN_MODE,
                "status": report["status"],
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            }
        )
        return
    if domain_backbone is not None:
        if "baseline" not in checkpoint_roots:
            raise FileNotFoundError(
                "A GONG backbone was attached without the baseline detector checkpoints"
            )
        print(
            {
                "mode": "pretrained-refiner-ablation",
                "checkpoint_root": str(checkpoint_roots["baseline"]),
                "domain_backbone": str(domain_backbone),
                "cuda_devices": list(range(torch.cuda.device_count())),
            }
        )
        report = run_pretrained_refiner_ablation(
            paths,
            checkpoint_roots["baseline"],
            domain_backbone,
            Path("/kaggle/working/pretrained_refiner_ablation"),
        )
        print(
            {
                "mode": "pretrained-refiner-ablation",
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
                "best_strategy": report["best_strategy"],
                "best_official_pq_delta": report["best_official_pq_delta"],
                "best_macro_pq_delta": report["best_macro_pq_delta"],
            }
        )
        return
    if set(checkpoint_roots) == {"baseline"}:
        print(
            {
                "mode": "baseline-oof-audit",
                "checkpoint_root": str(checkpoint_roots["baseline"]),
                "cuda_devices": list(range(torch.cuda.device_count())),
            }
        )
        report = run_baseline_audit(
            checkpoint_roots["baseline"],
            Path("/kaggle/working/baseline_audit"),
        )
        print(
            {
                "mode": "baseline-oof-audit",
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
                "best_postprocess": report["best_postprocess"]["variant"],
            }
        )
        return
    if checkpoint_roots:
        print(
            {
                "mode": "inference-only-recalibration",
                "checkpoint_roots": {
                    key: str(value) for key, value in checkpoint_roots.items()
                },
                "cuda_devices": list(range(torch.cuda.device_count())),
            }
        )
        report = run_recalibration(
            paths,
            cfg,
            checkpoint_roots,
            Path("/kaggle/working/recalibrated_ensembles"),
        )
        print(
            {
                "mode": report["mode"],
                "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
                "variants": list(report["variants"]),
            }
        )
        return
    validate_dual_gpu_refiner()
    with paths["json"].open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    indices = build_indices(coco)
    medoids = choose_medoid_records(indices)
    catalog = build_consensus_catalog(indices, medoids, cfg)
    observations = build_observations(indices)
    manifest = assign_folds(observations, cfg)
    work_dir = Path("/kaggle/working/spine_consensus")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(work_dir / "fold_manifest.csv", index=False)
    catalog_summary = {
        "physical_observations": len(observations),
        "annotator_records": len(coco["images"]),
        "raw_annotations": len(coco["annotations"]),
        "consensus_instances": sum(map(len, catalog.values())),
        "mean_consensus_instances": float(np.mean([len(value) for value in catalog.values()])),
        "debug": cfg.debug,
    }
    fingerprint = hashlib.sha256(
        (json.dumps(asdict(cfg), sort_keys=True) + manifest.to_csv(index=False)).encode()
    ).hexdigest()
    (work_dir / "run_config.json").write_text(
        json.dumps({"config": asdict(cfg), "catalog": catalog_summary, "fingerprint": fingerprint}, indent=2),
        encoding="utf-8",
    )
    print(catalog_summary)
    train_all(manifest, paths, catalog, indices, cfg, work_dir)
    run_oof(manifest, paths, indices, cfg, work_dir)
    submission = run_test(paths, cfg, work_dir)
    if not cfg.debug:
        test_ids = {path.stem for path in paths["test"].glob("*.jpeg")}
        submitted_ids = set(submission["filament_id"].str.rsplit("_", n=1).str[0])
        if submitted_ids != test_ids:
            raise RuntimeError(f"Submission misses {len(test_ids - submitted_ids)} test images")
        if submission.empty or not submission["filament_id"].is_unique:
            raise RuntimeError("Invalid submission rows")
    print(
        {
            "submission": str(work_dir / ("submission_debug.csv" if cfg.debug else "submission.csv")),
            "rows": len(submission),
            "debug": cfg.debug,
            "runtime_hours": round((time.monotonic() - run_started) / 3600, 3),
            "runtime_target_hours": cfg.runtime_target_hours,
        }
    )


if __name__ == "__main__":
    main()
