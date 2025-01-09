## Evaluation of the checkpoints with the lowest valid loss

# Point eval

## Raw temporal questions
python scripts/model/eval/point.py -m random -d temporal_questions -c raw
python scripts/model/eval/point.py -m majority -d temporal_questions -c raw
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -r 07efc5dbe75d2b5c7108832ca1b19619ed9200c0 -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -r e5219f261ff7b5809428a4e53b9946a5442474f8 -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -r 55b23176775dec273e2705b15e412f05a021a5db -d temporal_questions -c raw -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -r ad71005a0a3c3088079ec2103e62a07c7cf86f9e -d temporal_questions -c raw -b 1024

## Timeset
python scripts/model/eval/point.py -m random -d timeset
python scripts/model/eval/point.py -m majority -d timeset
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -r 07efc5dbe75d2b5c7108832ca1b19619ed9200c0 -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -r e5219f261ff7b5809428a4e53b9946a5442474f8 -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -r 55b23176775dec273e2705b15e412f05a021a5db -d timeset -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -r ad71005a0a3c3088079ec2103e62a07c7cf86f9e -d timeset -b 1024

## MATRES
python scripts/model/eval/point.py -m random -d matres
python scripts/model/eval/point.py -m majority -d matres
python scripts/model/eval/point.py -m hugosousa/smol-135-tq -r 07efc5dbe75d2b5c7108832ca1b19619ed9200c0 -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-augment -r e5219f261ff7b5809428a4e53b9946a5442474f8 -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-closure -r 55b23176775dec273e2705b15e412f05a021a5db -d matres -b 1024
python scripts/model/eval/point.py -m hugosousa/smol-135-tq-synthetic -r ad71005a0a3c3088079ec2103e62a07c7cf86f9e -d matres -b 1024

# Interval eval

## TempEval-3
python scripts/model/eval/interval.py -m random -d tempeval_3
python scripts/model/eval/interval.py -m majority -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -r 07efc5dbe75d2b5c7108832ca1b19619ed9200c0 -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -r e5219f261ff7b5809428a4e53b9946a5442474f8 -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -r 55b23176775dec273e2705b15e412f05a021a5db -d tempeval_3
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -r ad71005a0a3c3088079ec2103e62a07c7cf86f9e -d tempeval_3

## TDDiscourse
python scripts/model/eval/interval.py -m random -d tddiscourse
python scripts/model/eval/interval.py -m majority -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq -r 07efc5dbe75d2b5c7108832ca1b19619ed9200c0 -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-augment -r e5219f261ff7b5809428a4e53b9946a5442474f8 -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-closure -r 55b23176775dec273e2705b15e412f05a021a5db -d tddiscourse
python scripts/model/eval/interval.py -m hugosousa/smol-135-tq-synthetic -r ad71005a0a3c3088079ec2103e62a07c7cf86f9e -d tddiscourse
