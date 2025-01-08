export OMP_NUM_THREADS=$(nproc)

# Train on temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --config_file configs/classifier/smol-135.yaml
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 30
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 30

# Train on augmented temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30 --augment True
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 30 --augment True
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 30 --augment True

# Train on synthetic temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name synthetic_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name synthetic_temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 30
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name synthetic_temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 30

# Train on all temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name all_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-360M --dataset_name all_temporal_questions --batch_size 16 --gradient_accumulation_steps 8 --num_train_epochs 30
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-1.7B --dataset_name all_temporal_questions --batch_size 4 --gradient_accumulation_steps 32 --num_train_epochs 30

# Evaluate the classifiers
python scripts/model/eval.py -m random -d temporal_questions
python scripts/model/eval.py -m majority -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-TemporalQuestions -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-360M-TemporalQuestions -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d temporal_questions

python scripts/model/eval.py -m random -d timeset
python scripts/model/eval.py -m majority -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-TemporalQuestions -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-360M-TemporalQuestions -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-1.7B-TemporalQuestions -d timeset

python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-20250102151624 -d temporal_questions

# Debugging
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 50 --debug True
python scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 1 --do_train False --do_eval True --push_to_hub False

# Train on temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30

# Train on augmented temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30 --augment True

# Train on all temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name all_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30

# train on temporal questions augmented
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name all_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30 --augment True --early_stopping_patience 8

# Train on synthetic temporal questions
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name synthetic_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30 --early_stopping_patience 8

# train on synthetic temporal questions augmented
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py --model_name HuggingFaceTB/SmolLM2-135M --dataset_name synthetic_temporal_questions --batch_size 32 --gradient_accumulation_steps 4 --num_train_epochs 30 --augment True --early_stopping_patience 8

python scripts/model/eval.py -m models/smol-135-tq-false -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-temporal_questions-True -d temporal_questions -r e54bc0cf3c7afdf3edb3776c7fa1b86dd04bfce0
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-temporal_questions-False -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-synthetic_temporal_questions-True -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-synthetic_temporal_questions-False -d temporal_questions
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-True -d temporal_questions -r 623d65c5890a43b61088c9e4dd1455195013b835
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-False -d temporal_questions


python scripts/model/eval.py -m models/smol-135-tq-false -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-temporal_questions-True -d timeset -r e54bc0cf3c7afdf3edb3776c7fa1b86dd04bfce0
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-temporal_questions-False -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-synthetic_temporal_questions-True -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-synthetic_temporal_questions-False -d timeset
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-True -d timeset -r 623d65c5890a43b61088c9e4dd1455195013b835
python scripts/model/eval.py -m hugosousa/SmolLM2-135M-all_temporal_questions-False -d timeset
