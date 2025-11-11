#!/bin/bash
python unet/src/v1_train_unet.py --image-size 192 --batch-size 64 --epochs 50 --amp
