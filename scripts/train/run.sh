export OMP_NUM_THREADS=$(nproc)

# 135 model
CUDA_VISIBLE_DEVICES=0 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/raw.yaml
CUDA_VISIBLE_DEVICES=1 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/closure.yaml
CUDA_VISIBLE_DEVICES=2 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/augment.yaml
CUDA_VISIBLE_DEVICES=3 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/synthetic.yaml 
CUDA_VISIBLE_DEVICES=3 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/closure-augment.yaml
CUDA_VISIBLE_DEVICES=1 accelerate launch scripts/train/train.py --config_file configs/classifier/smol-135/closure-synthetic.yaml
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
