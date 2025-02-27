export OMP_NUM_THREADS=$(nproc)

## POINT MODELS
# 135 model
accelerate launch scripts/train/train.py --config_file configs/classifier/point/smol-135/raw.yaml 
accelerate launch scripts/train/train.py --config_file configs/classifier/point/smol-135/closure.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/point/smol-135/augment.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/point/smol-135/closure-augment.yaml

# 360M model
accelerate launch scripts/train/train.py --config_file configs/classifier/smol-360/raw.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/smol-360/closure.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/smol-360/augment.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/smol-360/closure-augment.yaml


## INTERVAL MODELS
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/debug.yaml 

# 135 model
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-135/raw.yaml 
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-135/closure.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-135/augment.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-135/closure-augment.yaml


# 360M model
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-360/raw.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-360/augment.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-360/closure.yaml
accelerate launch scripts/train/train.py --config_file configs/classifier/interval/smol-360/closure-augment.yaml
