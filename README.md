# Temporal Classifier

Train a model to classify temporal relations between temporal entities.

> Note: The codebase has been anonymized for the ACL 2025 submission. As a result, the HuggingFace model links are not publicly available, and the code examples below cannot be run as shown.

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

### Point-wise Evaluation

To get the tables below run:

```sh
sh scripts/eval/run.sh  # to run eval for each model
python scripts/utils/aggregate_results.py --relation_type point  # to aggregate the results
python scripts/utils/print_results.py --relation_type point  # to print the results
```

This table presents the macro average over the three label types for the TempEval-3 point-wise dataset.

| model    | inverse | closure | accuracy | precision | recall | f1-score |
| :------- | :------ | :------ | -------: | --------: | -----: | -------: |
| majority | ❌       | ❌       |    53.71 |      17.9 |  33.33 |     23.3 |
| random   | ❌       | ❌       |    33.96 |     33.47 |  34.25 |       30 |
| SmoLM2-135M | ❌       | ❌       |    77.21 |     69.12 |  65.91 |    67.18 |
|          | ✅       | ❌       |    78.39 |     70.23 |  69.49 |    69.82 |
|          | ❌       | ✅       |    79.49 |     69.41 |  70.18 |    69.78 |
|          | ✅       | ✅       |    80.22 |     74.23 |  63.75 |    66.14 |
| SmoLM2-360M | ❌       | ❌       |    80.65 |     71.38 |  70.51 |    70.92 |
|          | ✅       | ❌       |    81.32 |     73.46 |     71 |     72.1 |
|          | ❌       | ✅       |    82.13 |     75.71 |  68.27 |    70.65 |
|          | ✅       | ✅       |     82.1 |     74.96 |  71.48 |    72.86 |

### Interval Evaluation

#### SemEval Evaluation

To run the interval evaluation with the original SemEval script (available [here](https://github.com/naushadzaman/tempeval3_toolkit)) run the following steps.

```sh
sh scripts/utils/semeval.sh
```

This script will print the results presented in the table below.



| model                                                                            | I    | C    | $F_1$ |     P |     R |
| :------------------------------------------------------------------------------- | :--- | :--- | ----: | ----: | ----: |
| random                                                                           |      |      | 11.57 | 10.94 | 12.27 |
| majority                                                                         |      |      | 35.71 | 35.52 | 35.91 |
|                                                                                  |      |      |       |       |       |
| UTTime [link](https://aclanthology.org/S13-2015.pdf)                             |      |      | 56.45 | 55.58 | 57.35 |
| Graph Staking [link](https://www.jstage.jst.go.jp/article/jnlp/22/3/22_171/_pdf) |      |      | 57.78 | 57.63 | 57.92 |
| TRelPro [link](https://aclanthology.org/E14-1033.pdf)                            |      |      | 58.48 | 58.80 | 58.27 |
| CATENA [link](https://aclanthology.org/C16-1007/)                                |      |      |  61.9 |  62.6 |  61.3 |
| SP+ILP [link](https://aclanthology.org/D17-1108.pdf)                             |      |      |  67.2 |  69.1 |  65.5 |
|                                                                                  |      |      |       |       |       |
| Interval-135M                                                                     | ❌    | ❌    | 62.85 | 62.82 | 62.87 |
|                                                                                  | ✅    | ❌    | 64.93 | 64.89 | 64.97 |
|                                                                                  | ❌    | ✅    | 66.98 | 67.22 | 66.74 |
|                                                                                  | ✅    | ✅    | 66.22 | 66.48 | 65.97 |
|                                                                                  |      |      |       |       |       |
| Interval-360M                                                                     | ❌    | ❌    | 65.69 | 65.74 | 65.64 |
|                                                                                  | ✅    | ❌    | 68.98 | 69.01 | 68.95 |
|                                                                                  | ❌    | ✅    | 65.54 | 65.67 | 65.41 |
|                                                                                  | ✅    | ✅    | 35.42 | 35.06 | 35.80 |
|                                                                                  |      |      |       |       |       |
| IfP-135M                                                                         | ❌    | ❌    | 64.78 | 65.03 | 64.53 |
|                                                                                  | ✅    | ❌    | 63.88 | 63.68 | 64.09 |
|                                                                                  | ❌    | ✅    | 63.97 | 64.07 | 63.87 |
|                                                                                  | ✅    | ✅    | 64.01 | 64.37 | 63.65 |
|                                                                                  |      |      |       |       |       |
| IfP-360M                                                                         | ❌    | ❌    | 66.24 | 66.18 | 66.30 |
|                                                                                  | ✅    | ❌    | 70.12 | 70.19 | 70.06 |
|                                                                                  | ❌    | ✅    | 67.91 | 68.08 | 67.73 |
|                                                                                  | ✅    | ✅    | 69.28 | 69.39 | 69.17 |
