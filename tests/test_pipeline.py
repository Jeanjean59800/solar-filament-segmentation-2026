"""Light tests of source fidelity, compressed RLE and ensemble invariants.

These load only NumPy, SciPy and pycocotools; no torch, images or inference.
"""
from __future__ import annotations

import ast
import json
import math
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source-notebooks' / 'filament-v23-hybrid-sweep.py'
PURE_NAMES = {
    'FusionPolicy', 'AuditSetting', 'ProposalV2Setting', 'ProposalV3Setting',
    'mask_iou', 'resolve_overlaps', 'cluster_representative', 'cluster_fold_predictions',
    'resolve_fused_overlaps', 'fuse_predictions_with_policy', 'encode_mask',
    'predictions_for_proposal_setting', 'predictions_for_v3_setting', 'audit_candidate_indices',
}
TREE = ast.parse((ROOT / 'pipeline.py').read_text())
PURE_TREE = ast.Module(body=[node for node in TREE.body if getattr(node, 'name', None) in PURE_NAMES], type_ignores=[])
exec(compile(PURE_TREE, str(ROOT / 'pipeline.py'), 'exec'), globals())


def rectangle(x0=5, x1=25, y0=5, y1=25):
    result = np.zeros((40, 40), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


class SourceFidelityTests(unittest.TestCase):
    def test_every_extracted_definition_matches_the_executed_source(self):
        source_tree = ast.parse(SOURCE.read_text())
        original = {getattr(node, 'name', None): node for node in source_tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        extracted = [node for node in TREE.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        self.assertGreater(len(extracted), 25)
        for node in extracted:
            with self.subTest(definition=node.name):
                self.assertEqual(ast.dump(node, include_attributes=False),
                                 ast.dump(original[node.name], include_attributes=False))

    def test_no_training_functions_in_inference_module(self):
        self.assertFalse(any(getattr(node, 'name', '').startswith('train_') for node in TREE.body))


class PipelineInvariantTests(unittest.TestCase):
    def test_coco_compressed_rle_round_trip(self):
        for mask in (rectangle(), ~rectangle(), np.zeros((40, 40), bool), np.ones((40, 40), bool)):
            text = encode_mask(mask)
            decoded = mask_utils.decode({'size': list(mask.shape), 'counts': text.encode('ascii')})
            np.testing.assert_array_equal(decoded.astype(bool), mask)

    def test_same_fold_duplicates_do_not_inflate_support(self):
        mask = rectangle()
        clusters = cluster_fold_predictions([(0, mask, 0.9), (0, mask, 0.8), (1, mask, 0.7)], 0.2)
        self.assertTrue(all(len({item[0] for item in cluster}) == len(cluster) for cluster in clusters))
        self.assertEqual(max(len(cluster) for cluster in clusters), 2)

    def test_three_fold_medoid_is_an_input_mask(self):
        masks = [rectangle(), rectangle(6, 26), rectangle(7, 27)]
        predictions = [(fold, mask, 0.9) for fold, mask in enumerate(masks)]
        fused, _, detail = fuse_predictions_with_policy(predictions, FusionPolicy('test', 0.2, 3, 'medoid', 1))
        self.assertEqual(len(fused), 1)
        self.assertFalse(detail['fallback'])
        np.testing.assert_array_equal(fused[0], masks[1])

    def test_low_support_fallback_is_recorded(self):
        masks, _, detail = fuse_predictions_with_policy([(0, rectangle(), 0.9)], FusionPolicy('test', 0.2, 3, 'medoid', 1))
        self.assertEqual(len(masks), 1)
        self.assertTrue(detail['fallback'])

    def test_resolved_instances_do_not_overlap(self):
        masks, _ = resolve_overlaps([rectangle(), rectangle(15, 35)], [0.9, 0.8], 1)
        self.assertEqual(len(masks), 2)
        self.assertFalse(np.logical_and(*masks).any())

    def test_v3_expected_quality_filter(self):
        record = {'seed': rectangle(), 'bounds': (5, 5, 25, 25),
                  'probability': np.ones((20, 20)), 'detector_score': 0.7,
                  'existence': 0.8, 'predicted_iou': 0.8}
        rejected = dict(record, existence=0.9, predicted_iou=0.2)
        masks, _ = predictions_for_v3_setting([record, rejected], ProposalV3Setting(0.55, 0.3, 0.5), 96)
        self.assertEqual(len(masks), 1)

    def test_seed_selection_preserves_low_confidence_second_pass(self):
        records = [{'seed': rectangle(), 'detector_score': 0.08}]
        self.assertEqual(audit_candidate_indices(records, AuditSetting('test', detector_conf=0.2)), [0])

    def test_checkpoint_manifest_contains_final_fold_assignment(self):
        manifest = json.loads((ROOT / 'weights' / 'manifest.json').read_text())
        paths = {entry['path'] for entry in manifest['files']}
        expected = {f'detector_fold{fold}.pt' for fold in range(5)}
        expected |= {f'refiner_v3_fold{fold}.pt' for fold in range(3)}
        expected |= {f'refiner_v2_fold{fold}.pt' for fold in (3, 4)}
        self.assertEqual(paths, expected)
        self.assertTrue(all(entry['bytes'] < 80 * 1024 * 1024 for entry in manifest['files']))


if __name__ == '__main__':
    unittest.main()
