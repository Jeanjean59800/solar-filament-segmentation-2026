#!/usr/bin/env python3
"""Check strict checkpoint/architecture compatibility and synthetic output shapes."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from ultralytics import YOLO
    import pipeline as p
    from inference import verify_weights
    weights = ROOT / 'weights'
    verify_weights(weights)
    torch.set_num_threads(2)
    results = []
    for fold in range(5):
        detector = YOLO(str(weights / f'detector_fold{fold}.pt'))
        if detector.task != 'segment':
            raise RuntimeError('Expected a segmentation detector')
        architecture = 'v3' if fold < 3 else 'v2'
        model = p.ProposalRefinerV3() if fold < 3 else p.ProposalRefinerV2()
        state = torch.load(weights / f'refiner_{architecture}_fold{fold}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            tensor = torch.zeros((1, 3, 64, 64))
            outputs = model(tensor, torch.zeros((1, 6))) if fold < 3 else model(tensor)
        assert tuple(outputs[0].shape) == (1, 1, 64, 64)
        assert all(tuple(value.shape) == (1,) for value in outputs[1:])
        results.append({'fold': fold, 'architecture': architecture, 'detector_task': detector.task,
                        'output_shapes': [list(value.shape) for value in outputs]})
    print(json.dumps({'status': 'strict_load_and_shapes_passed', 'models': results}, indent=2))

if __name__ == '__main__':
    main()
