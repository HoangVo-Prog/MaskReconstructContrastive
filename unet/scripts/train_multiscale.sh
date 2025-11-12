#!/bin/bash
VER=$1

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align
