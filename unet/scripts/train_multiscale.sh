#!/bin/bash
python unet/src/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align
