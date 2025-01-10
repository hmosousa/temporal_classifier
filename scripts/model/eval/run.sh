# Point eval

## Raw temporal questions
python scripts/model/eval/point.py -m random -d temporal_questions -c raw
python scripts/model/eval/point.py -m majority -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d temporal_questions -c raw

## Closure temporal questions
python scripts/model/eval/point.py -m random -d temporal_questions -c closure
python scripts/model/eval/point.py -m majority -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d temporal_questions -c closure
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d temporal_questions -c closure

## Timeset
python scripts/model/eval/point.py -m random -d timeset
python scripts/model/eval/point.py -m majority -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d timeset

## MATRES
python scripts/model/eval/point.py -m random -d matres
python scripts/model/eval/point.py -m majority -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d matres

# Interval eval

## TempEval-3
python scripts/model/eval/interval.py -m random -d tempeval_3
python scripts/model/eval/interval.py -m majority -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tempeval_3

## TDDiscourse
python scripts/model/eval/interval.py -m random -d tddiscourse
python scripts/model/eval/interval.py -m majority -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tddiscourse
