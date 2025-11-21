#!/bin/bash
VER=$1
ADNI_PATH=$2
IMAGE_TYPE=$3  # axial or coronal

if [ "$USE_MASKED_LOSS" = "true" ]; then
  MASKED_FLAG="--enable-masked-loss"
else
  MASKED_FLAG=""
fi

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --adni True --adni-path "$ADNI_PATH" --image-type "$IMAGE_TYPE" \
  $MASKED_FLAG 
