export OMP_NUM_THREADS=$(nproc)

# Train on temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 10

# Train on synthetic temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name synthetic_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name synthetic_temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name synthetic_temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 10

# Train on all temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name all_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name all_temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 10
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name all_temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 10

# Evaluate the classifiers
python scripts/model/eval.py -m hugosousa/SmolLM-135M-TemporalQuestions -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-20250102151624 -d temporal_questions


# Debugging
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 1 --debug True
python scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 1 --do_train False --do_eval True --push_to_hub False
