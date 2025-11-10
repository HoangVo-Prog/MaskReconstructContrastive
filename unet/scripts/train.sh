#!/bin/bash
python unet/src/train_ssl_unet.py --image-size 192 --batch-size 64 --epochs 50 --amp
