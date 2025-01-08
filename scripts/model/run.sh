export OMP_NUM_THREADS=$(nproc)

# Train
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --config_file configs/classifier/smol-135.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --config_file configs/classifier/smol-135-augment.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --config_file configs/classifier/smol-135-closure.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --config_file configs/classifier/smol-135-synthetic.yaml


# Eval
python scripts/model/eval.py -m models/smol-135-tq -d temporal_questions -b 1024
python scripts/model/eval.py -m models/smol-135-tq-augment -d temporal_questions -b 1024
python scripts/model/eval.py -m models/smol-135-tq-closure -d temporal_questions -b 1024
python scripts/model/eval.py -m models/smol-135-tq-synthetic -d temporal_questions -b 1024

python scripts/model/eval.py -m models/smol-135-tq -d timeset -b 1024
python scripts/model/eval.py -m models/smol-135-tq-augment -d timeset -b 1024
python scripts/model/eval.py -m models/smol-135-tq-closure -d timeset -b 1024
python scripts/model/eval.py -m models/smol-135-tq-synthetic -d timeset -b 1024
