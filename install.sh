#!/bin/bash

conda create -n vggt python=3.11
conda activate vggt

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install e .
cd ..

pip install viser opencv-python
