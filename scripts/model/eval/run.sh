# Point eval
python scripts/model/eval/point.py -m models/smol-135-tq -d temporal_questions -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-augment -d temporal_questions -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-closure -d temporal_questions -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-synthetic -d temporal_questions -b 1024

python scripts/model/eval/point.py -m models/smol-135-tq -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-augment -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-closure -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-synthetic -d timeset -b 1024

# Interval eval
python scripts/model/eval/interval.py -m random -d tempeval_3
python scripts/model/eval/interval.py -m majority -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-augment -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-closure -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-synthetic -d tempeval_3
