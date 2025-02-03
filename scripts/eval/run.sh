# Point eval

# Best models
# smol-135
#   - raw: 
#   - augment: hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
#   - closure: hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
#   - synthetic: 
#   - closure-augment: hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c
#   - closure-synthetic: 
#   - augment-synthetic: 
#   - closure-augment-synthetic: 

# smol-360
#   - raw: hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
#   - augment: hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5
#   - closure: 
#   - synthetic: 
#   - closure-augment: 
#   - closure-synthetic: 

# smol-1.7
#   - raw: 
#   - augment: 
#   - closure: 
#   - synthetic: 
#   - closure-augment: 
#   - closure-synthetic: 

## Point Tempeval
python scripts/eval/point.py -d point_tempeval -m random 
python scripts/eval/point.py -d point_tempeval -m majority 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6  # augment
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33  # closure
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-tq-synthetic 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-tq-closure-synthetic 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-tq-augment-synthetic 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-tq-closure-augment-synthetic 

python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba 
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5 
# Interval eval

## TempEval-3
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m random
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m majority 

python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135 
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-tq-synthetic 
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-tq-closure-synthetic 
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-tq-augment-synthetic 
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-135-tq-closure-augment-synthetic 

python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5 



python scripts/eval/interval.py -d interval_tempeval -s most_likely -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5 
python scripts/eval/interval.py -d interval_tddiscourse -s most_likely -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5 
python scripts/eval/interval.py -d interval_timebank_dense -s most_likely -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5 

# With the original semeval script
python scripts/eval/semeval_eval.py -m random
python scripts/eval/semeval_eval.py -m majority
python scripts/eval/semeval_eval.py -m hugosousa/smol-135
python scripts/eval/semeval_eval.py -m hugosousa/smol-135-a

python scripts/eval/semeval_eval.py -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba




########### Other

## Timeset
python scripts/eval/point.py -m random -d timeset
python scripts/eval/point.py -m majority -d timeset
python scripts/eval/point.py -m hugosousa/smol-135 -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-a -d timeset
python scripts/eval/point.py -m hugosousa/smol-135-7fd02948 -d timeset
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
python scripts/eval/point.py -m hugosousa/smol-135-7fd02948 -d matres
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
python scripts/eval/point.py -m hugosousa/smol-135-7fd02948 -d point_tddiscourse
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
python scripts/eval/point.py -m hugosousa/smol-135-7fd02948 -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-augment-synthetic -d point_timebank_dense
python scripts/eval/point.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d point_timebank_dense


## TDDiscourse
python scripts/eval/interval.py -m random -d tddiscourse
python scripts/eval/interval.py -m majority -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135 -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-a -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-7fd02948 -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d tddiscourse
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d tddiscourse

## Timebank-Dense
python scripts/eval/interval.py -m random -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m majority -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135 -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-a -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-7fd02948 -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-augment-synthetic -d interval_timebank_dense -s most_likely
python scripts/eval/interval.py -m hugosousa/smol-135-tq-closure-augment-synthetic -d interval_timebank_dense -s most_likely
