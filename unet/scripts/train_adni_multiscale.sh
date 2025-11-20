#!/bin/bash
VER=$1
ADNI_PATH=$2
IMAGE_TYPE=$3  # axial or coronal

python unet/src/"$VER"/train.py --amp \
  --use-gn --use-se --use-multiscale \
  --pre-bias --pre-norm --pre-crop --pre-align \
  --adni --adni-path "$ADNI_PATH" --image-type "$IMAGE_TYPE"
