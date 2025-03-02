# Point eval

# Best models
### Point
# smol-135
#   - raw: hugosousa/smol-135-0dd0da37 -r aadb93fe6ee0272e40eac187511de316afa94b5b
#   - augment: hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
#   - closure: hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
#   - closure-augment: hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c

# smol-360
#   - raw: hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
#   - augment: hugosousa/smol-360-a-4a820490 -r ccfb712dc239a3ae22513808fd6863a7a135b044
#   - closure: hugosousa/smol-360-c-e82ebef2 -r 9a78bbf916620cba266c9c718061a6ff370bad05
#   - closure-augment: hugosousa/smol-360-ac-b19ae776 -r 04ad7dc726ee0bc5f067cbf659033dda36de34da


### Interval

# smol-135
#   - raw: hugosousa/smol-135-interval-1b7d11c1 -r 4ae68e8121803a5a32f6883cb55bd8d5ef777cc5
#   - augment: hugosousa/smol-135-interval-a-6ba0463e -r 25e1cd390e80e0c712dddad4ea44851b9ad3c1fa
#   - closure: hugosousa/smol-135-interval-c-7a430df7 -r d88979a7f06bf9e55fcdcd01bc82f0f7a791cb97
#   - closure-augment: hugosousa/smol-135-interval-ca-86f6ae17 -r ffdb27fdaee3666dd57500329f82b60d370d54b7

# smol-360
#   - raw: hugosousa/smol-360-interval-575aff8f -r 3101569dbc5f4ceb80114f603f0263c291c89cf7
#   - augment: hugosousa/smol-360-interval-a-04fbd03a -r b36899397b68cf475ba010fb6fc070828b6203eb
#   - closure: hugosousa/smol-360-interval-c-6bd44a78 -r 3bfa8b35600a2a5feabd571c2f0e341999c87581
#   - closure-augment: hugosousa/smol-360-interval-ca-8bd7cf03 -r 7efb562fa7d8015b2624052c7c818b12624e23ca



## Point Tempeval
python scripts/eval/point.py -d point_tempeval -m random 
python scripts/eval/point.py -d point_tempeval -m majority 

python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-0dd0da37 -r aadb93fe6ee0272e40eac187511de316afa94b5b
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c

python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-a-4a820490 -r ccfb712dc239a3ae22513808fd6863a7a135b044
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-c-6af17138 -r e5924e499d8076c6117e5cd38edc21c75f95b7ac
python scripts/eval/point.py -d point_tempeval -m hugosousa/smol-360-ac-b19ae776 -r 5c44cd7d5950f2c24a19a59144e15a4be1817f91

# Interval eval

## TempEval-3
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m random
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m majority 

# point system
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-0dd0da37 -r aadb93fe6ee0272e40eac187511de316afa94b5b
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c

python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-a-4a820490 -r ccfb712dc239a3ae22513808fd6863a7a135b044 
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-c-e82ebef2 -r 9a78bbf916620cba266c9c718061a6ff370bad05
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-ac-b19ae776 -r 5c44cd7d5950f2c24a19a59144e15a4be1817f91

# interval model
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-1b7d11c1 -r 4ae68e8121803a5a32f6883cb55bd8d5ef777cc5
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-a-6ba0463e -r 25e1cd390e80e0c712dddad4ea44851b9ad3c1fa
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-c-7a430df7 -r d88979a7f06bf9e55fcdcd01bc82f0f7a791cb97
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-ca-86f6ae17 -r ffdb27fdaee3666dd57500329f82b60d370d54b7

python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-575aff8f -r 3101569dbc5f4ceb80114f603f0263c291c89cf7
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-a-04fbd03a -r b36899397b68cf475ba010fb6fc070828b6203eb
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-c-6bd44a78 -r 3bfa8b35600a2a5feabd571c2f0e341999c87581
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-ca-8bd7cf03 -r 7efb562fa7d8015b2624052c7c818b12624e23ca


# Calibration

python scripts/eval/calibration.py -m hugosousa/smol-135-0dd0da37 -r aadb93fe6ee0272e40eac187511de316afa94b5b
python scripts/eval/calibration.py -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
python scripts/eval/calibration.py -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
python scripts/eval/calibration.py -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c

python scripts/eval/calibration.py -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
python scripts/eval/calibration.py -m hugosousa/smol-360-a-4a820490 -r ccfb712dc239a3ae22513808fd6863a7a135b044
python scripts/eval/calibration.py -m hugosousa/smol-360-c-6af17138 -r e5924e499d8076c6117e5cd38edc21c75f95b7ac
python scripts/eval/calibration.py -m hugosousa/smol-360-ac-b19ae776 -r 5c44cd7d5950f2c24a19a59144e15a4be1817f91
