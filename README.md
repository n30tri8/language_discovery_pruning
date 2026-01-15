**incomplete readme**

## Dependecies
pyton 3.10
hf download Qwen/Qwen2.5-7B-Instruct --local-dir .\raw_model\
benchmarks, see benchamrks section
## Installation 🛠️

git clone https://github.com/n30tri8/language_discovery_pruning.git
COPY raw_model/ and ./benchmark_data inside project directory
set env vars:
    HF_TOKEN="hf_..."
    RUN_ENV=prod_os
    CUDA_VISIBLE_DEVICES=0,1
edit this variable to point to the persistent disk to save pruned models
    run_env['model_dir'] = "/mnt/povobackup/clic/p.torabi"
create venv
activate: source venv/bin/activate
pip install -r requirements.txt

## Usage:
```bash
python main.py --model meta-llama/Llama-3.1-8B-Instruct [--test_num N] [--sparsity_ratios RATIOS] [--languages LANGS]
```
examples
1. Pruning without saving the pruned model 
   python main.py --model Qwen/Qwen2.5-7B-Instruct --run prune --sparsity_ratios 35 50 60  --languages en hi it --no-save_pruned
2. Evaluting raw model on MMLU 
   python main.py --model meta-llama/Llama-3.1-8B-Instruct --run raw_eval --test_num 300
3. prune and then evaluate pruned model 
   python main.py --model meta-llama/Llama-3.1-8B-Instruct --run prune cross_eval --test_num 300 --sparsity_ratios 60


## Benchmarks 📊
MMMLU
GLUE
XGLUE
uinauil, textual entailment section, for italian
it translated
arabic paraphrase
indicparahprase

## Acknowledgements 🙏
This project builds upon two key repositories:
* [WANDA](https://github.com/locuslab/wanda) - For the core pruning methodology
The code has been significantly adapted and streamlined for our specific experiments, removing unused components and simplifying the evaluation pipeline.
* initiation from pruning on TOM, changed heavily including the pruning to match wanda and 

## License 📜
This project is licensed under the MIT License - see the `LICENSE` file for detail