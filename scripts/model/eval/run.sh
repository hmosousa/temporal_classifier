# Point eval

## Raw temporal questions
python scripts/model/eval/point.py -m random -d temporal_questions -c raw
python scripts/model/eval/point.py -m majority -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d temporal_questions -c raw -b 1024

python scripts/model/eval/point.py -m hugosousa/smol-360-tq -d temporal_questions -c raw -b 1024

python scripts/model/eval/point.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d temporal_questions -c raw -b 256

## Timeset
python scripts/model/eval/point.py -m random -d timeset
python scripts/model/eval/point.py -m majority -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d timeset -b 1024

## MATRES
python scripts/model/eval/point.py -m random -d matres
python scripts/model/eval/point.py -m majority -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d matres -b 1024

# Interval eval

## TempEval-3
python scripts/model/eval/interval.py -m random -d tempeval_3
python scripts/model/eval/interval.py -m majority -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tempeval_3

python scripts/model/eval/interval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d tempeval_3

## TDDiscourse
python scripts/model/eval/interval.py -m random -d tddiscourse
python scripts/model/eval/interval.py -m majority -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tddiscourse

python scripts/model/eval/interval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d tddiscourse
