# Assignment 1
# Natural Language Processing
# Steven Glass

List of packages

  - python=3.12
  - ipykernel
  - jupyterlab
  - matplotlib
  - numpy
  - pandas
  - re
  - unicodedata



## How to run this code

Main reproducible environment:
- `environment.yml`

You must create and activate the environment from the YAML file as follows

```bash
conda env create -f environment.yml
conda activate nlp-assign1
```

### Run Trigram test

python -m scripts.run_trigram --file_name data/train.en.txt --seed 42

### Run BPE test

