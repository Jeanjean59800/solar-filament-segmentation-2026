#!/usr/bin/env python3
"""Extract the inference dependency closure verbatim from the recovered V23 run."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source-notebooks' / 'filament-v23-hybrid-sweep.py'
ROOTS = {
    'predict_proposal_v2_instances', 'predict_proposal_v3_instances',
    'fuse_predictions_with_policy', 'encode_mask', 'seed_everything',
    'validate_submission_frame', 'legacy_inference_cfg',
}
HEADER = '''from __future__ import annotations

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

'''

def main():
    source = SOURCE.read_text()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node
    selected = set(ROOTS)
    while True:
        dependencies = {
            node.id for name in selected for node in ast.walk(nodes[name])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in nodes
        }
        expanded = selected | dependencies
        if expanded == selected:
            break
        selected = expanded
    blocks = []
    for name in sorted(selected, key=lambda name: nodes[name].lineno):
        node = nodes[name]
        first_line = min([node.lineno] + [item.lineno for item in getattr(node, 'decorator_list', [])])
        blocks.append(''.join(lines[first_line-1:node.end_lineno]).rstrip())
    content = HEADER + '\n\n\n'.join(blocks) + '\n'
    (ROOT / 'pipeline.py').write_text(content)
    (ROOT / 'provenance' / 'extraction.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'selected_definitions': sorted(selected),
        'pipeline_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'changes_to_selected_definitions': 'none; imports and an inference CLI are separate',
    }, indent=2) + '\n')
    print(f'Extracted {len(selected)} definitions into pipeline.py')

if __name__ == '__main__':
    main()
