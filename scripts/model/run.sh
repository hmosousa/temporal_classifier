export OMP_NUM_THREADS=$(nproc)



# Train the classifiers
accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --gradient_accumulation_steps 4 \ 
    scripts/model/train.py \
    --model_name HuggingFaceTB/SmolLM2-135M \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10

accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --batch_size 16
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --batch_size 4

# Evaluate the classifiers
python scripts/model/eval.py -m hugosousa/SmolLM-135M-TemporalQuestions -d temporal_questions
