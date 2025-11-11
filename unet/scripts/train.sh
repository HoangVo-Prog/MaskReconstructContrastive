#!/bin/bash
python unet/src/train.py --image-size 192 --batch-size 64 --epochs 50 --amp
