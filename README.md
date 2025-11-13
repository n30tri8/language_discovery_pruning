# help to run
docker build
docker tag us-west1-docker.pkg.dev/sonorous-treat-476419-s8/main-docker/pruning:latest
docker push us-west1-docker.pkg.dev/sonorous-treat-476419-s8/main-docker/pruning:latest

gcloud ai custom-jobs create `
  --region=us-west1 `
  --display-name="de_raw_prune_eval_try_1" `
  --config=vertex_job_config.yaml

#### checklist of running
- model
- dataset samples in templates
- LINGUISTIC_BENCHMARKS, comment\uncomment
- machinespec in vertex ai
- job name



# TODO
- [x] evaluate_model_on_dataset is buggy
- [x] evaluate raw model on all subjects and report restlus
### for real test
- [x] update subjects
- [x] update benchmarks for real test, update train counts and test counts, both glue and mmlu
- [x] update model and run
- [x] uncomment both procedures in main.py
### load model problem
- [x] download in local and upload to gcs
- [x] should I return low_cpu_mem_usage=True
- [x] try different machine: n1-standard-4 with two gpu or n1-standard-8 with one gpu
### next optimization
- [x] profile the next run
- [] use persistent disk for saving pruned models: https://cloud.google.com/compute/docs/disks#disk-types
- [x] test: save model on anoterh thread
- [x] recheck models, I think I copies the wrong folder of meta-llama
- [x] rebuild docker image after fixing the above issues
- [] download from windows made twice the size of saved model


### German GLUE
- [x] update SELECTED_XGLUE_TASKS sample counts, change model
- [x] test in prod
- [x] omit cross-lingual from SELECTED_XGLUE_TASKS prompts
- [x] xglue prompts in german
- [x] mmmlue in german
  - [x] check\enforce same question across all languages
    phil(en, de), professional_law(en, de; first 550), high_school_mathematics(en, de), professional_psychology(en, de)
  - [x] promting in german
- add another section of glue, or mmlu??
- rerun en pruning on the same xglue?


### more robust run
- selection of :
  - sample_size
  - sparse ratio
  - machine for running: https://cloud.google.com/vertex-ai/docs/training/configure-compute?_gl=1*5c5yis*_ga*MTc1NTkxODIzNC4xNzQ1MzQyMzU2*_ga_WH2QY8WWF5*czE3NjIxMTMzNDQkbzQwJGcxJHQxNzYyMTE0MTQ5JGo0JGwwJGgw#specifying_gpus
    -     machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 2
    - n1-standard-16 ; mem 60
    - g2-standard-12; mem 48
    -> e2-highmem-8 ; mem 64
    - n2-highmem-8; ; mem 64
    - n1-highmem-8; mem 52
    - for mem, ~45 is enough
  - section of mmlu
  - sections of xglue


### useful features
- [x] separation of different parts in prune()
- selection of which parts to run, instead of commenting different parts
- add this or fork
This project is a fork of <original repo>.
Currently maintained by @yourname.
- [] add these to cmd: model to run, languages to run


dowload model through:
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir .\raw_model\





# Language Sparsity & Cognition Project 🧠

This repository contains the code and experiments for our paper ["Pruning for Mindreading: Analyzing Theory of Mind Capabilities in Sparse Language Models"](pruning_for_mindreading.pdf).

## Overview 📝

This project explores how model pruning affects Theory of Mind (ToM) capabilities in Large Language Models. We use targeted pruning on specific ToM tasks to analyze how sparsity impacts performance across different mindreading challenges.

## Project Structure 📂

The codebase is focused on a streamlined evaluation pipeline that:
- Loads pretrained language models
- Prunes them using task-specific calibration data  
- Evaluates ToM capabilities on multiple benchmarks
- Generates detailed performance analysis across sparsity levels

## Installation 🛠️

1. Create a new conda environment:
```bash
conda create -n prune_on_tom python=3.10
conda activate prune_on_tom
```
2. Install required packages:
```
pip install -r requirements.txt
```

## Usage 🚀

The main script (`main.py`) supports various parameters for model evaluation and pruning:

```bash
python main.py [--models MODEL_NAMES] [--train_num N] [--test_num N] [--sparsity_ratios RATIOS] [--seed SEED]
```

Parameters:
- `--models`: List of model names/paths (default: meta-llama/Llama-3.2-3b-Instruct)
- `--train_num`: Calibration set size per subtask (default: 32)
- `--test_num`: Test set size per subtask (default: 5, use negative for all)
- `--sparsity_ratios`: List of pruning percentages (default: 25 50 75)
- `--seed`: Random seed (default: 42)

Example usage:
```bash
# Run with default parameters
python main.py

# Run with custom models and sparsity ratios
python main.py --models meta-llama/Llama-2-7b-chat tiiuae/falcon-7b --sparsity_ratios 30 60 90

# Use larger calibration and test sets
python main.py --train_num 64 --test_num 20
```

## Data 📊
All evaluation data is sourced from the ToMBench repository and can be found in the `data/` folder.
This includes various Theory of Mind tasks like:
* False Belief Task
* Faux-pas Recognition
* Hinting Task
* Strange Stories Task
* And more...

## Acknowledgements 🙏
This project builds upon two key repositories:
* [WANDA](https://github.com/locuslab/wanda) - For the core pruning methodology
* [ToMBench](https://github.com/zhchen18/ToMBench) - For the Theory of Mind evaluation framework
The code has been significantly adapted and streamlined for our specific experiments, removing unused components and simplifying the evaluation pipeline.

## License 📜
This project is licensed under the MIT License - see the `LICENSE`file for detail