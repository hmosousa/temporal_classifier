export OMP_NUM_THREADS=$(nproc)

# Train the classifiers
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --batch_size 16

# Evaluate the classifiers
python scripts/model/eval.py -m hugosousa/SmolLM-135M-TemporalQuestions -d temporal_questions
