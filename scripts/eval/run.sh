# Point eval

## Tempeval
python scripts/eval/point.py -m random -d point_tempeval
python scripts/eval/point.py -m majority -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135 -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-a -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d point_tempeval
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d point_tempeval

## Timeset
python scripts/eval/point.py -m random -d timeset
python scripts/eval/point.py -m majority -d timeset
python scripts/eval/point.py -m hugosousa/smol-135 -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-a -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d timeset

## MATRES
python scripts/eval/point.py -m random -d matres
python scripts/eval/point.py -m majority -d matres
python scripts/eval/point.py -m hugosousa/smol-135 -d matres
python scripts/eval/point.py -m hugosousa/smol-135-a -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d matres
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d matres

## Point TDDiscourse
python scripts/eval/point.py -m random -d point_tddiscourse
python scripts/eval/point.py -m majority -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135 -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-a -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d point_tddiscourse
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d point_tddiscourse

## Point TimeBank-Dense
python scripts/eval/point.py -m random -d point_timebank_dense
python scripts/eval/point.py -m majority -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135 -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-a -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d point_timebank_dense

# Interval eval

## TempEval-3
python scripts/eval/interval.py -m random -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m majority -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135 -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-a -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d interval_tempeval -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d interval_tempeval -s most_likely

## Timebank-Dense
python scripts/eval/interval.py -m random -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m majority -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135 -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-a -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d interval_timebank_dense -s most_likely

## TDDiscourse
python scripts/eval/interval.py -m random -d tddiscourse
python scripts/eval/interval.py -m majority -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135 -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-a -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tddiscourse
