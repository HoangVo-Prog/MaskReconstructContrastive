#!/bin/bash
python unet/src/train.py --amp \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --tsne-every 5
