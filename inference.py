#!/usr/bin/env python3
"""Reproduce V23 hybrid012_s3_medoid from the ten supplied checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_SHA256 = '58b3a7c52c239d309189537636e5b0f76b1a8c2dc67d72bb6eb0338eb18677a1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(weights: Path) -> dict:
    manifest = json.loads((weights / 'manifest.json').read_text())
    for entry in manifest['files']:
        path = weights / entry['path']
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Checkpoint integrity failure: {path}')
    if len(manifest['files']) != 10:
        raise RuntimeError('The final ensemble requires exactly ten checkpoint files')
    return manifest


def environment() -> dict:
    packages = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'ultralytics-thop', 'numpy',
                 'pandas', 'opencv-python', 'scipy', 'pycocotools'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    weights = args.weights.resolve()
    manifest = verify_weights(weights)
    if args.verify_only:
        result = {'status': 'weights_verified', 'files': len(manifest['files']),
                  'bytes': sum(item['bytes'] for item in manifest['files'])}
        print(json.dumps(result, indent=2))
        return result

    # Heavy dependencies are imported only for actual image inference.
    import pandas as pd
    import torch
    from ultralytics import YOLO
    import pipeline as p

    if args.device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count() != args.require_gpus:
            raise RuntimeError(f'Requested exactly {args.require_gpus} CUDA devices; '
                               f'found {torch.cuda.device_count()}. No device fallback is used.')
        devices = [torch.device(f'cuda:{fold % args.require_gpus}') for fold in range(5)]
    else:
        devices = [torch.device('cpu')] * 5

    if not args.images or not args.images.is_dir():
        raise FileNotFoundError('--images must be the official test/test_images directory')
    images = sorted(args.images.glob('*.jpeg'))
    if len(images) != args.expected_images:
        raise RuntimeError(f'Expected {args.expected_images} JPEG files, found {len(images)}')
    if args.limit_images is not None:
        if args.limit_images < 1:
            raise ValueError('--limit-images must be positive')
        images = images[:args.limit_images]

    p.seed_everything(p.CFG().seed)
    v2_cfg = p.legacy_inference_cfg()
    v2_cfg.refiner_crop, v2_cfg.refiner_min_crop = 384, 128
    v2_cfg.seed_nms_iou, v2_cfg.minimum_area = 0.45, 96
    v3_cfg = p.legacy_inference_cfg()
    v3_cfg.detector_conf = 0.05
    v3_cfg.refiner_crop, v3_cfg.refiner_min_crop, v3_cfg.refiner_batch = 448, 128, 4
    v3_cfg.seed_nms_iou, v3_cfg.minimum_area = 0.45, 96
    v2_setting = p.ProposalV2Setting(0.50, 0.50)
    v3_setting = p.ProposalV3Setting(0.55, 0.30, 0.50)
    policy = p.FusionPolicy('s3_medoid', 0.20, 3, 'medoid', minimum_area=96)
    models = {}
    for fold in range(5):
        device = devices[fold]
        detector = YOLO(str(weights / f'detector_fold{fold}.pt'))
        architecture = 'v3' if fold < 3 else 'v2'
        model = p.ProposalRefinerV3() if architecture == 'v3' else p.ProposalRefinerV2()
        model.load_state_dict(torch.load(weights / f'refiner_{architecture}_fold{fold}.pt',
                                         map_location=device, weights_only=True), strict=True)
        models[fold] = (detector, model.to(device).eval(), device)

    rows, diagnostics = [], []
    for number, image_path in enumerate(images, 1):
        predictions = []
        # Preserve the original V23 prediction ordering: V2 folds then V3 folds.
        for fold in (3, 4, 0, 1, 2):
            detector, model, device = models[fold]
            if fold < 3:
                masks, scores = p.predict_proposal_v3_instances(
                    image_path, detector, model, device, v3_cfg, v3_setting)
            else:
                masks, scores = p.predict_proposal_v2_instances(
                    image_path, detector, model, device, v2_cfg, v2_setting)
            predictions.extend((fold, mask, score) for mask, score in zip(masks, scores))
        masks, _, detail = p.fuse_predictions_with_policy(predictions, policy)
        for index, mask in enumerate(masks, 1):
            rows.append({'filament_id': f'{image_path.stem}_{index}',
                         'segmentation_rle': p.encode_mask(mask)})
        diagnostics.append({'filename': image_path.name, 'raw_instances': len(predictions),
                            'instances': len(masks), 'fallback': bool(detail['fallback']),
                            'support_histogram': json.dumps(detail['support_histogram'], sort_keys=True)})
        print(json.dumps({'image': number, 'images': len(images), 'rows': len(rows)}), flush=True)

    submission = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    p.validate_submission_frame(submission, images, 'V23 hybrid012_s3_medoid')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    pd.DataFrame(diagnostics).to_csv(args.output.with_suffix('.diagnostics.csv'), index=False)
    output_hash = sha256(args.output)
    complete = len(images) == 180 and args.expected_images == 180
    result = {'status': 'complete' if complete else 'smoke_only', 'variant': 'hybrid012_s3_medoid',
              'output': str(args.output), 'sha256': output_hash, 'rows': len(rows),
              'images': len(images), 'fallback_images': sum(item['fallback'] for item in diagnostics),
              'historical_reference_sha256': REFERENCE_SHA256,
              'matches_historical_reference': output_hash == REFERENCE_SHA256 if complete else None,
              'runtime_seconds': time.monotonic() - started, 'environment': environment(),
              'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              'weight_manifest_sha256': sha256(weights / 'manifest.json'),
              'pipeline_sha256': sha256(ROOT / 'pipeline.py'),
              'inference_sha256': sha256(Path(__file__).resolve())}
    args.output.with_suffix('.run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if args.check_reference and (not complete or output_hash != REFERENCE_SHA256):
        raise RuntimeError('Output differs from the historical reference; inspect the recorded runtime and diagnostics')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights')
    parser.add_argument('--images', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'submission.csv')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--require-gpus', type=int, default=2,
                        help='Exact CUDA device count; historical V23 used two T4 GPUs')
    parser.add_argument('--expected-images', type=int, default=180)
    parser.add_argument('--limit-images', type=int, help='Explicit smoke run, without a full-reproduction claim')
    parser.add_argument('--verify-only', action='store_true', help='Verify all checkpoint hashes; no ML dependencies needed')
    parser.add_argument('--check-reference', action='store_true', help='Fail if the full CSV SHA256 differs from V23')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
