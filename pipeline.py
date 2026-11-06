from __future__ import annotations

"""V23 inference definitions, extracted verbatim by scripts/build_core.py.

No training code runs on import. See ATTRIBUTION.md and provenance/source.json.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

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


@dataclass(frozen=True)
class AuditSetting:
    name: str
    detector_conf: float = 0.20
    seed_nms_iou: float = 0.60
    refiner_threshold: float = 0.65
    minimum_area: int = 96
    morphology: str = "none"


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def encode_mask(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


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


V3_METADATA_DIM = 6


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


def validate_submission_frame(
    submission: pd.DataFrame, test_files: Sequence[Path], label: str
) -> None:
    test_ids = {path.stem for path in test_files}
    submitted_ids = set(submission["filament_id"].str.rsplit("_", n=1).str[0])
    if submitted_ids != test_ids:
        raise RuntimeError(f"{label} misses {len(test_ids - submitted_ids)} test images")
    if submission.empty or not submission["filament_id"].is_unique:
        raise RuntimeError(f"Invalid {label} candidate")
