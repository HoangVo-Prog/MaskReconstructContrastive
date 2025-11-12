#!/bin/bash
# GroupNorm + SE for high-level stability
VER=$1

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se \
  --pre-bias --pre-norm --pre-crop --pre-align
