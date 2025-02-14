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
#   - augment: hugosousa/smol-135-interval-a-4fad7bc3 -r 8da557a8fe7776cf89e704fb287a00c7ad5400c1
#   - closure: hugosousa/smol-135-interval-c-eda01c25 -r bf8fcba0f7401596549083bd04c1b79ae1517031
#   - closure-augment: hugosousa/smol-135-interval-ca-87e1b0c9 -r e90271b51a444e12b196097b858765431d7f37dc

# smol-360
#   - raw: hugosousa/smol-360-interval-df83a28e -r 4db6d51a2010ef53e92189e7f8058c497d80a850
#   - augment: hugosousa/smol-360-interval-a-5f554f47 -r ec9ee1dd470e9ecbe66dd4b2c45b6ad6713c3d30
#   - closure: hugosousa/smol-360-interval-c-74c05ab6 -r 9ee5c9afb819f8ca2402a9c35d0b31dc9441427e
#   - closure-augment: hugosousa/smol-360-interval-ca-737f2a4a -r 64b6e96c7ab83a5a415b53afb9968d4ab0151543



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
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-a-4fad7bc3 -r 8da557a8fe7776cf89e704fb287a00c7ad5400c1
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-c-eda01c25 -r bf8fcba0f7401596549083bd04c1b79ae1517031
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-135-interval-ca-87e1b0c9 -r e90271b51a444e12b196097b858765431d7f37dc

python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-df83a28e -r 4db6d51a2010ef53e92189e7f8058c497d80a850
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-a-5f554f47 -r ec9ee1dd470e9ecbe66dd4b2c45b6ad6713c3d30
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-c-74c05ab6 -r 9ee5c9afb819f8ca2402a9c35d0b31dc9441427e
python scripts/eval/interval.py -d interval_tempeval -strategy most_likely -m hugosousa/smol-360-interval-ca-737f2a4a -r 64b6e96c7ab83a5a415b53afb9968d4ab0151543
