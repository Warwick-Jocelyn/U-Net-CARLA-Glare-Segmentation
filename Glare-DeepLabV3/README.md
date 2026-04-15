# Semantic Segmentation with the CARLA simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alt text](https://img.shields.io/pypi/pyversions/python-binance.svg)


Python scripts to perform semantic segmentation of driving images generated with the CARLA simulator, making use of a pytorch implementation of [DeepLabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/). The notebooks included are:

- `generate_semantic_segmentation_data.ipynb`: CARLA python script to generate semantic segmentation dataset, creating RGB images and semantic masks.

- `semantic_segmentation_carla.ipynb`: train DeepLabv3 model for semantic segmentation in CARLA images.

Sample data included in `CARLA_smalldataset.tar.xz`.
