python scripts/utils/aggregate_results.py --relation_type point
python scripts/utils/aggregate_results.py --relation_type interval

python scripts/utils/print_results.py --relation_type point
python scripts/utils/print_results.py --relation_type interval

python scripts/utils/graphs/results_bars.py --metric f1-score
python scripts/utils/graphs/results_bars.py --metric accuracy
python scripts/utils/graphs/results_bars.py --metric precision
python scripts/utils/graphs/results_bars.py --metric recall
