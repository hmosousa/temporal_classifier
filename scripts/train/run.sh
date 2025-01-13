export OMP_NUM_THREADS=$(nproc)

# 135 model
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/raw.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/closure.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/synthetic.yaml 
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/closure-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/closure-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/augment-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-135/closure-augment-synthetic.yaml


# 360M model
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/raw.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/closure.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/closure-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/closure-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/augment-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-360/closure-augment-synthetic.yaml



# 1.7B model
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/raw.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/closure.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/closure-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/closure-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/augment-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/train/train.py --config_file configs/classifier/smol-1.7/closure-augment-synthetic.yaml
