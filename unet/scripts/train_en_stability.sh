#!/bin/bash
# GroupNorm + SE for high-level stability
VER=$1

if [ "$USE_MASKED_LOSS" = "true" ]; then
  MASKED_FLAG="--enable-masked-loss"
else
  MASKED_FLAG=""
fi

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se \
  --pre-bias --pre-norm --pre-crop --pre-align \
  $MASKED_FLAG