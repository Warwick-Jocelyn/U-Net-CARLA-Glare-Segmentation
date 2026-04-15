# U-Net / DeepLabV3 — CARLA Glare Semantic Segmentation

Semantic segmentation of driving-scene images rendered in the [CARLA](https://carla.org/) simulator, with a focus on **scenes containing camera glare / lens flare** (e.g. strong sun, headlights, wet-road reflections). Two architectures are trained and compared on the same dataset:

- **U-Net** — a lightweight encoder–decoder baseline trained from scratch.
- **DeepLabV3 (ResNet-50)** — a stronger backbone fine-tuned from COCO pre-trained weights.

The goal is to study how glare affects per-class segmentation quality on autonomous-driving-style inputs, and to provide a reproducible training / evaluation pipeline on a small CARLA-generated dataset.

---

## Repository layout

```
.
├── Carla.ipynb                     # End-to-end notebook: CARLA dataset → U-Net training
├── Carla_Glare.ipynb               # Same pipeline, restricted to glare scenes
├── Carla2CityScapesPalette.py      # Remap CARLA class IDs → Cityscapes colour palette
├── bitcheck.py                     # Utility: inspect bit-depth / class IDs of mask PNGs
│
├── Glare-U-Net/                    # U-Net implementation
│   ├── train.py                    # Training loop (AMP, tqdm, train/val split)
│   ├── eval.py                     # Per-class IoU / accuracy evaluation
│   ├── test.py                     # Inference on a test split
│   ├── demo.py                     # Run trained model on demo images
│   ├── rename_delete.py            # Dataset housekeeping
│   └── losses.png                  # Training loss curve
│
└── Glare-DeepLabV3/                # DeepLabV3 (ResNet-50) implementation
    ├── semantic_segmentation_carla.ipynb        # Train / evaluate DeepLabV3 on CARLA
    ├── generate_semantic_segmentation_data.ipynb # CARLA client script to render RGB + masks
    ├── train.py
    ├── Demo_Data/                  # Sample input images and ground-truth labels
    ├── losses.png
    └── LICENSE
```

## Classes

The CARLA semantic-segmentation sensor labels are mapped to the following 13 classes:

| ID | Class       | RGB colour       |
|----|-------------|------------------|
| 0  | Unlabeled   | (0, 0, 0)        |
| 1  | Building    | (70, 70, 70)     |
| 2  | Fence       | (100, 40, 40)    |
| 3  | Other       | (55, 90, 80)     |
| 4  | Pedestrian  | (220, 20, 60)    |
| 5  | Pole        | (153, 153, 153)  |
| 6  | Roadline    | (157, 234, 50)   |
| 7  | Road        | (128, 64, 128)   |
| 8  | Sidewalk    | (244, 35, 232)   |
| 9  | Vegetation  | (107, 142, 35)   |
| 10 | Car         | (0, 0, 142)      |
| 11 | Wall        | (102, 102, 156)  |
| 12 | Traffic sign| (220, 220, 0)    |

`Carla2CityScapesPalette.py` additionally provides a Cityscapes-style colour mapping for cross-dataset visualisation.

---

## Environment

Tested on Linux + CUDA with PyTorch. Minimum dependencies:

```bash
pip install torch torchvision opencv-python numpy matplotlib \
            scikit-learn tqdm pillow torchsummary
```

For dataset generation you will additionally need a running [CARLA simulator](https://carla.org/) and the matching `carla` Python client.

---

## Dataset

The training data consists of RGB frames and semantic masks rendered from the CARLA simulator, split into a *general* set and a *glare* subset. Paths inside the scripts (e.g. in `train.py`) point to local directories and should be adjusted to match your environment:

```python
image_path = ["/path/to/CameraRGB/"]
mask_path  = ["/path/to/CameraSeg/"]
```

A small sample dataset for DeepLabV3 ships as `CARLA_smalldataset.tar.xz` (hosted on Google Drive — see below).

---

## Pre-trained weights & sample data

The trained checkpoints and the sample dataset archive are **not** stored in this repository because they exceed GitHub's 100 MB per-file limit. They can be downloaded from Google Drive:

| File | Model | Size | Link |
|------|-------|------|------|
| `Unet_epoch_100_checkpoint.pth`        | U-Net (epoch 100)            | 119 MB | *TODO: paste Google Drive link* |
| `Unet_epoch_98_checkpoint.pth`         | U-Net (epoch 98)             | 119 MB | *TODO: paste Google Drive link* |
| `final.pth`                             | U-Net (final)                | 119 MB | *TODO: paste Google Drive link* |
| `deeplabv3_resnet50_coco-cd0a2569.pth` | DeepLabV3 COCO pre-trained   | 161 MB | *TODO: paste Google Drive link* |
| `CARLA_small_dataset_resnet50_*.pt`    | DeepLabV3 trained on CARLA   | 161 MB | *TODO: paste Google Drive link* |
| `Glare_Dataset_resnet50_*.pt`          | DeepLabV3 trained on Glare set | 161 MB | *TODO: paste Google Drive link* |
| `CARLA_smalldataset.tar.xz`            | Sample dataset               | —      | *TODO: paste Google Drive link* |

After downloading, place the `.pth` / `.pt` files back under the corresponding `Glare-U-Net/` or `Glare-DeepLabV3/` directory.

---

## Quick start

### U-Net

```bash
cd Glare-U-Net
# edit image_path / mask_path in train.py, then:
python train.py            # train
python eval.py             # per-class IoU / accuracy
python demo.py             # run inference on demo images
```

### DeepLabV3

```bash
cd Glare-DeepLabV3
# open and run:
jupyter notebook semantic_segmentation_carla.ipynb
# or use:
python train.py
```

---

## Results

Training loss curves are saved as `losses.png` inside each model directory. Both models are trained without heavy data augmentation (`dataaug_False`, `batch_size=2`) so that the glare vs. non-glare comparison stays controlled.

---

## License

The DeepLabV3 sub-project is released under the MIT License (see `Glare-DeepLabV3/LICENSE`). The rest of the repository is provided for research and educational purposes.

## Acknowledgements

- [CARLA Simulator](https://carla.org/)
- [PyTorch DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- U-Net: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015.
