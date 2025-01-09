export OMP_NUM_THREADS=$(nproc)

# 135 model
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-closure.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-synthetic.yaml 
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-closure-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-closure-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-augment-synthetic.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train/train.py --config_file configs/classifier/smol-135-closure-augment-synthetic.yaml
