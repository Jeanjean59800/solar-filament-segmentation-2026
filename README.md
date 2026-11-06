# Solar filament segmentation: V23 proposal ensemble

This repository packages our V23 `hybrid012_s3_medoid` solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). It includes the source, a complete inference notebook, and all ten trained checkpoints. Inference starts from official JPEG images and pretrained weights; it does not train models or read stored predictions.

The final ensemble combines five YOLO11s segmentation detectors, V3 proposal refiners for folds 0–2, and V2 proposal refiners for folds 3–4. One-to-one cross-fold association, support from three folds, and a medoid mask produce the final instances. The approach extends the public YOLO/refiner work of ektarr (Maxim); see [ATTRIBUTION.md](ATTRIBUTION.md).

The four-page technical report and its official-template sources are in [report/](report/). This package is published at https://github.com/Jeanjean59800/solar-filament-segmentation-2026.

## Results and evidence

| Quantity | Historical V23 value | Evidence |
|---|---:|---|
| Public leaderboard PQ, rounded | 0.39 | Kaggle submission reference `55607242` |
| Strict-OOF reference PQ | 0.43520179088181965 | Recovered V23 experiment report |
| Official test images | 180 | Recovered experiment report |
| Predicted instances | 1,173 | Original output hash and report |
| Consensus fallback images | 1 | Recovered experiment report |

The OOF value is a historical reference recorded by the experiment, not a fresh validation replay. Model and threshold choices used validation experiments. The public score is a rounded public leaderboard result, with no private-score claim. The all-V3 variant scored 0.38; the all-V2 method also scored 0.39. A separate external HDJojo control recorded 0.55 and is not this model or this repository's claimed result.

The historical output SHA-256 is:

```text
58b3a7c52c239d309189537636e5b0f76b1a8c2dc67d72bb6eb0338eb18677a1
```

The reference CSV and official images/annotations are not redistributed. The inference program never reads a reference CSV. [provenance/reference.json](provenance/reference.json) and [provenance/historical-v23-report.json](provenance/historical-v23-report.json) preserve the output identity and historical results.

## Run from pretrained weights

Use Python 3.12 and two CUDA GPUs. The historical run used **two Tesla T4 GPUs**, Python 3.12.13, PyTorch 2.10.0+cu128 and Ultralytics 8.4.103. Each of those three versions is attested in the recovered baseline log. The V23 metadata also records the immutable Kaggle container digest in [provenance/environment.json](provenance/environment.json).

The full historical package freeze was not saved. [requirements.txt](requirements.txt) supplies exact direct pins for a declared Python 3.12 porting target. Its other numerical-library versions are selected target versions, rather than asserted historical observations. The nine target package versions and torchvision/PyTorch compatibility were checked against primary package metadata. A complete installation and full GPU numerical replay remain to be verified for this release.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inference.py --verify-only
```

Download the official competition data with your own Kaggle account after accepting its rules, then extract it outside this repository. Set the image path to the resulting `MAGFiLO_1.0_Kaggle_2026/test/test_images` directory. Training images and labels are not needed by the inference command.

```bash
python inference.py \
  --images /path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --output outputs/submission.csv \
  --device cuda --require-gpus 2 --check-reference
```

The command requires exactly two visible CUDA devices and has no silent device fallback. It writes `submission.csv`, `submission.diagnostics.csv` and `submission.run.json`. The JSON records the actual environment, GPU names, source and model-manifest hashes, image and instance counts, and comparison with the historical CSV hash. `--check-reference` fails on a full-output mismatch while preserving these diagnostics. Different hardware or numerical packages can change threshold-adjacent masks, so a mismatch must be investigated before claiming exact reproduction.

For an explicit remote smoke run, add `--limit-images 2` and omit `--check-reference`. This checks only two images and marks the result `smoke_only`. It does not establish full reproduction. Open [reproduce_v23.ipynb](reproduce_v23.ipynb) to run the same image-to-CSV pipeline interactively.

The historical four-variant V23 sweep took 1.42 hours on two T4s. That is historical sweep timing, not a measured runtime for the single-variant command above.

## Checkpoint inventory

[weights/manifest.json](weights/manifest.json) lists the SHA-256, byte length, fold, source Kaggle kernel and original output path for each file. The ten checkpoints total 390,700,051 bytes; every individual file is below 80 MiB and can be stored in ordinary Git without splitting or Git LFS.

| Local checkpoint | Source |
|---|---|
| `detector_fold0.pt` … `detector_fold4.pt` | `jeanjean111/filament-baseline-ektarr` |
| `refiner_v3_fold0.pt` … `refiner_v3_fold2.pt` | `jeanjean111/filament-proposal-v3-full-resume` |
| `refiner_v2_fold3.pt`, `refiner_v2_fold4.pt` | `jeanjean111/filament-proposal-refiner-v2` |

The refiners instantiate ResNet-18 with `weights=None`, then strictly load the supplied trained state dictionaries. They do not download initialization weights. Detectors load their trained `.pt` checkpoints.

## Exact inference policy

The detector runs at 1024 pixels, confidence 0.05, box-NMS IoU 0.80, class-agnostic NMS and native-size masks. Mask-level seed NMS uses IoU 0.45. V2 selects up to 20 proposals at detector confidence 0.20; V3 selects up to 24 at confidence 0.05. If that selection is empty, the historical low-confidence second pass is preserved.

Crops use context 1.8 and minimum native side 128. The three channels are grayscale, CLAHE grayscale, and the binary seed. V2 crops are 384 pixels with four flip views; V3 crops are 448 pixels with identity and horizontal-flip views. V2 requires quality at least 0.50. V3 requires existence at least 0.55 and existence × predicted mask IoU at least 0.30. Both use mask threshold 0.50 and minimum area 96 pixels.

Cross-fold Hungarian association uses mask IoU 0.20 with at most one vote per fold and the historical fold ordering. Ordinary clusters need three folds. Medoid selection retains an actual member mask. Overlap cleanup retains masks with at least 96 pixels and at least 65% of their original area. The strongest-cluster fallback is preserved when no ordinary consensus candidate survives. Output masks use compressed COCO RLE in native image coordinates. Final V23 refiners have a single mask output; they do not predict a separate spine channel.

## Source and validation

[pipeline.py](pipeline.py) contains 33 inference definitions extracted verbatim from the recovered [V23 source script](source-notebooks/filament-v23-hybrid-sweep.py). [scripts/build_core.py](scripts/build_core.py) reproduces the extraction. The original script also preserves training and validation code for inspection; it is not the inference entry point. The recovered [baseline notebook](source-notebooks/baseline/filament-baseline-ektarr.ipynb) documents the detector lineage.

The release preparation verified all ten checkpoint hashes and passed ten lightweight tests covering source-definition fidelity, compressed RLE round trips, fold-vote uniqueness, medoid selection, overlap cleanup, V3 quality filtering, fallback behavior and final fold assignment. These tests do not run trained-model image inference.

```bash
python -m unittest discover -s tests -v
python scripts/verify_models.py
```

The second command strictly loads every checkpoint and checks synthetic refiner output shapes. It is supplied for the target environment and was not executed on the local preparation host, which has no PyTorch installation. A fresh end-to-end GPU result is currently pending; checkpoint recovery and algorithm tests alone do not establish numerical reproduction of the historical score.
