**incomplete readme**

## Installation 🛠️

1. Create a new conda environment:
```bash
conda create -n language_discovery_pruning python=3.10
conda activate language_discovery_pruning
```
2. Install required packages:
```
pip install -r requirements.txt
```

## Usage 🚀
```bash
python main.py --model meta-llama/Llama-3.1-8B-Instruct [--test_num N] [--sparsity_ratios RATIOS] [--languages LANGS]
```

Parameters:
- `--model`: Model name/path (eg: meta-llama/Llama-3.2-3b-Instruct)
- `--test_num`: Test set size per subtask (default: 50, use negative for all)
- `--sparsity_ratios`: List of pruning percentages (default: 50)
- `--languages`: One or more language codes from the built-in benchmarks (default: all available)

## Data 📊
MMMLU
GLUE
XGLUE

## Acknowledgements 🙏
This project builds upon two key repositories:
* [WANDA](https://github.com/locuslab/wanda) - For the core pruning methodology
The code has been significantly adapted and streamlined for our specific experiments, removing unused components and simplifying the evaluation pipeline.

## License 📜
This project is licensed under the MIT License - see the `LICENSE` file for detail