# Point eval

## Raw temporal questions
python scripts/eval/point.py -m random -d temporal_questions --config-name raw
python scripts/eval/point.py -m majority -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d temporal_questions --config-name raw -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d temporal_questions --config-name raw
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d temporal_questions --config-name raw

## Closure temporal questions
python scripts/eval/point.py -m random -d temporal_questions --config-name closure
python scripts/eval/point.py -m majority -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d temporal_questions --config-name closure -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d temporal_questions --config-name closure
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d temporal_questions --config-name closure

## Timeset
python scripts/eval/point.py -m random -d timeset
python scripts/eval/point.py -m majority -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d timeset -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d timeset

## MATRES
python scripts/eval/point.py -m random -d matres
python scripts/eval/point.py -m majority -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d matres -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d matres

# Interval eval

## TempEval-3
python scripts/eval/interval.py -m random -d tempeval_3
python scripts/eval/interval.py -m majority -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure -d tempeval_3 -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tempeval_3
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tempeval_3

## TDDiscourse
python scripts/eval/interval.py -m random -d tddiscourse
python scripts/eval/interval.py -m majority -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure -d tddiscourse -r eb226f2e2f317d559463b7be943936196debc033
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tddiscourse
