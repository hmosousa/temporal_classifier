# Point eval

## Raw temporal questions
python scripts/model/eval/point.py -m random -d temporal_questions -c raw
python scripts/model/eval/point.py -m majority -d temporal_questions -c raw
python scripts/model/eval/point.py -m models/smol-135-tq -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-augment -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-closure -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-synthetic -d temporal_questions -c raw -b 1024

python scripts/model/eval/point.py -m models/smol-360-tq -d temporal_questions -c raw -b 1024

python scripts/model/eval/point.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d temporal_questions -c raw -b 256

## Timeset
python scripts/model/eval/point.py -m random -d timeset
python scripts/model/eval/point.py -m majority -d timeset
python scripts/model/eval/point.py -m models/smol-135-tq -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-augment -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-closure -d timeset -b 1024
python scripts/model/eval/point.py -m models/smol-135-tq-synthetic -d timeset -b 1024

# Interval eval

## TempEval-3
python scripts/model/eval/interval.py -m random -d tempeval_3
python scripts/model/eval/interval.py -m majority -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-augment -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-closure -d tempeval_3
python scripts/model/eval/interval.py -m models/smol-135-tq-synthetic -d tempeval_3

python scripts/model/eval/interval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d tempeval_3

## TDDiscourse
python scripts/model/eval/interval.py -m random -d tddiscourse
python scripts/model/eval/interval.py -m majority -d tddiscourse
python scripts/model/eval/interval.py -m models/smol-135-tq -d tddiscourse

python scripts/model/eval/interval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d tddiscourse
