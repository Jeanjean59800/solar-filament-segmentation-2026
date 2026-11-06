# Attribution and provenance

The trained baseline was adapted from **ektarr (Maxim)**, especially [New Metric | Solar Filament](https://www.kaggle.com/code/ektarr/new-metric-solar-filament) and the earlier [YOLO + U-Net | Solar Filament Segmentation](https://www.kaggle.com/code/ektarr/yolo-u-net-solar-filament-segmentation). The YOLO detector plus seed-conditioned ResNet/U-shaped refiner pattern is credited to that public baseline. Our trained derivative is `jeanjean111/filament-baseline-ektarr`.

The proposal V2 keep/reject head, V3 existence correction and predicted-mask-IoU head, associated training/calibration experiments, and V23 hybrid/fusion selection are part of our subsequent competition work. Existing backbone architectures are not claimed as new contributions. The release entry point and extraction/checking scripts package the recovered implementation for inference from saved weights.

The final source was recovered from `jeanjean111/filament-v23-hybrid-sweep`. Its exact source hash and model-source mappings are in `provenance/source.json` and `weights/manifest.json`. These kernel references identify our original run lineage; the repository includes the files needed for final inference even if a kernel is private or later changes.

The implementation uses the Ultralytics YOLO11 software/model family, PyTorch, torchvision ResNet-18, OpenCV, NumPy, SciPy, pandas and the COCO mask API. These dependencies retain their respective licenses. Ultralytics software and pretrained model licensing must be retained when redistributing its derivatives; consult the license supplied with the exact dependency version.

Official GONG/MAGFiLO competition images and annotations remain with their original providers and are not included here. Access them through the [official competition](https://www.kaggle.com/competitions/filament-segmentation-2026). The final task-specific training used the official competition training labels. No external release of test labels is used by this package.

The 0.55 external HDJojo control and a derivative submission are separate provenance records. Their source model artifact was not recovered for a complete reproducible release, and neither is represented by this V23 package or its 0.39 own-method claim.
