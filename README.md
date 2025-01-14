# Temporal Classifier

Train an agent to classify temporal relations.

## Setup

Create `.env` file with the following:

```
HF_TOKEN=<your-huggingface-token>
HF_USERNAME=<your-huggingface-username>
GOOGLE_API_KEY=<your-google-api-key>  # for gemini
```

For users:

```sh
conda create -p ./.conda python=3.11
conda activate ./.conda
pip install -e .
```

For developers:

```sh
conda create -p ./.conda python=3.11
conda activate ./.conda
pip install poetry
poetry install
poetry run pre-commit install
```

The developer setup installs Poetry for dependency management, installs all project dependencies, and sets up pre-commit hooks to maintain code quality and consistency across the project.

## Training

```sh
accelerate config
export OMP_NUM_THREADS=$(nproc)
accelerate launch --config_file configs/accelerate/zero2.yaml scripts/model/train.py 
```

### Profile the code

```sh
python -m cProfile -o profile.prof main.py
snakeviz profile.prof
```

## Results

### Classifier

To get the tables below run:

```sh
python scripts/utils/aggregate_results.py
python scripts/utils/print_results.py
```

This table presents the macro average over the four label types for the temporal questions (TQ) and the timeset (TS) datasets. 

R: raw
A: augmented
S: synthetic


| model        | R    | A    | S    | TQ   |       Acc |     $F_1$ | TS     |       Acc |     $F_1$ |
| :----------- | :--- | :--- | :--- | :--- | --------: | --------: | :----- | --------: | --------: |
| random       | ✓    |      |      |      |     33.07 |     29.38 |        |      33.7 |     32.41 |
| majority     | ✓    |      |      |      |     53.71 |      23.3 |        |     41.32 |     19.49 |
|              |      |      |      |      |           |           |        |           |           |
| SmolLM2-135M | ✓    |      |      |      |     66.33 | **53.66** |        |     41.32 |     34.63 |
|              | ✓    | ✓    |      |      |     61.41 |     38.33 |        |     34.05 |     31.46 |
|              | ✓    |      | ✓    |      | **67.71** |      42.2 |        |     41.94 | **37.61** |
|              | ✓    | ✓    | ✓    |      | **67.71** |     40.34 |        |     41.77 |     37.21 |
|              |      |      | ✓    |      |     43.22 |     27.29 |        |     39.64 |     23.59 |
|              |      | ✓    | ✓    |      |     34.58 |     23.54 |        | **46.32** |      25.5 |
|              |      |      |      |      |           |           |        |           |           |
| SmolLM2-360M | ✓    |      |      |      | **72.28** |     45.43 |        | **48.57** | **42.26** |
|              |      |      |      |      |           |           |        |           |           |
| SmolLM2-1.7B | ✓    |      |      |      |     71.96 | **60.37** |        |     46.88 |     41.38 |


## Load Models from Hugging Face

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "hugosousa/classifier_llama_1b", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("hugosousa/classifier_llama_1b")

inputs = tokenizer(["Hello, world!"], return_tensors="pt")
outputs = model(**inputs)
```

or using the pipeline

```python
import torch
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="hugosousa/classifier_llama_1b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(classifier(["Hello, world!"]))

```


## Demo Application

To lunch the app run:

```sh
python app/app.py
```
