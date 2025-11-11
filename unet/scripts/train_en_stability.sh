#!/bin/bash
# GroupNorm + SE for high-level stability
python unet/src/train_ssl_unet.py --amp \
  --use-gn --use-se \
  --pre-bias --pre-norm --pre-crop --pre-align
