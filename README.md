# Language-Specific Pruning of Large Language Models
(TODO: make project intro more precise)

This project investigates the effects of language-specific pruning on the performance of multilingual Large Language Models (LLMs). 
We use the Wanda pruning technique to sparsify a model based on calibration data from a specific language and then evaluate its performance on both general knowledge (MMLU) and linguistic capability benchmarks for that language.

The core workflow consists of three main procedures:
-   **`raw_eval`**: Evaluates the original, unpruned model on a subset of MMLU benchmarks to establish a performance baseline.
-   **`prune`**: Prunes the model using language-specific calibration data (e.g., from XGLUE) at various sparsity ratios. It also evaluates the pruned model's linguistic capabilities on the same benchmark it was pruned on.
-   **`cross_eval`**: Performs a cross-evaluation by testing the language-pruned models on MMLU benchmarks to assess how language-specific pruning affects general knowledge in that language.

The project supports experiments across several languages using the following benchmarks:
-   **Linguistic Benchmarks**: Used for pruning calibration and linguistic evaluation.
    -   `en`, `de`, `fr`: Using tasks from the XGLUE benchmark.
    -   `it`, `ar`, `hi`: Using a combination of custom sentence entailment and paraphrase datasets.
-   **General Knowledge Benchmarks**: Used for baseline and cross-evaluation.
    -   A selection of subjects from the MMLU benchmark, including `philosophy`, `international_law`, `high_school_mathematics`, and more.

## Dependencies
*   Python 3.10
*   A Llama-style Large Language Model compatible with the Wanda pruning codebase (e.g., `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`).
*   PyTorch and other libraries listed in `requirements.txt`.
*   Benchmark datasets (see [Benchmarks](#benchmarks-📊) section).

To download a model, you can use `huggingface-cli`:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./raw_model/
```

## Installation 🛠️

### Local Environment
1.  Clone the repository.

2.  Place the pre-downloaded model into a `raw_model/` directory and benchmark data into a `benchmark_data/` directory within the project root.

3.  Set required environment variables.
    ```bash
    HF_TOKEN="hf_..."
    RUN_ENV="prod_os" 
    # Optional: specify GPUs
    # CUDA_VISIBLE_DEVICES=0,1
    ```
    *Note: `RUN_ENV` determines storage paths. `prod_os` is recommended for local setups where you can specify a persistent disk.*

4.  If using `RUN_ENV=prod_os`, edit `main.py` to set the path for storing pruned models:
    ```python
    # in main.py
    elif runtime_env_arg == 'prod_os':
        run_env['root_storage_dir'] = project_dir
        run_env['model_dir'] = "/path/to/your/persistent/storage/" # EDIT THIS
    ```

5.  Create and activate a virtual environment:

6.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Google Cloud (Vertex AI)
This project can be run as a [custom job on Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job). This requires setting up a GCP project, a GCS bucket for logs and models, and an Artifact Registry.

1.  **Build and Push Docker Image:**
    The provided `Dockerfile` is a multi-stage build that packages the model and dependencies for efficient deployment.
    ```bash
    # Build the image
    docker build -t us-west1-docker.pkg.dev/{project-id}/{artifact-registry}/pruning:latest .

    # Push to Artifact Registry
    docker push us-west1-docker.pkg.dev/{project-id}/{artifact-registry}/pruning:latest
    ```

2.  **Configure and Run Job:**
    Edit the `vertex_job_config.yaml` to specify your machine type and other settings. Then, create the custom job:
    ```bash
    gcloud ai custom-jobs create \
      --region=us-west1 \
      --display-name="my-pruning-job" \
      --config=vertex_job_config.yaml
    ```

## Usage
The main script `main.py` is controlled via command-line arguments.

```bash
python main.py --model <model_name> [options]
```

### Arguments
-   `--model`: (Required) The Hugging Face model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
-   `--run`: The procedures to execute. Choose one or more from `raw_eval`, `prune`, `cross_eval`. (Default: `raw_eval`).
-   `--sparsity_ratios`: A list of pruning percentages (e.g., `35 50 60`). (Default: `50`).
-   `--languages`: A list of language codes to run experiments on (e.g., `en hi it`). (Default: all available languages).
-   `--test_num`: Number of samples to use from the test set for evaluations. (Default: `50`).
-   `--save_pruned` / `--no-save_pruned`: Flag to enable/disable saving of pruned models. (Default: enabled).
-   `--seed`: Random seed for reproducibility. (Default: `42`).

### Examples
1.  **Evaluate the raw (unpruned) model on MMLU for all languages:**
    ```bash
    python main.py --model Qwen/Qwen2.5-7B-Instruct --run raw_eval --test_num 300
    ```

2.  **Prune a model on English, Hindi, and Italian data without saving the artifacts:**
    ```bash
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --run prune --sparsity_ratios 35 50 60 --languages en hi it --no-save_pruned
    ```

3.  **Prune and then cross-evaluate on all available languages:**
    ```bash
    python main.py --model meta-llama/Llama-3.1-8B-Instruct --run prune cross_eval --test_num 300 --sparsity_ratios 60
    ```

## Adding a New Language
To extend the experiment to a new language, follow these steps:

1.  **Add Benchmark Data**:
    -   Provide calibration/evaluation data, ideally sentence pair tasks like entailment (XNLI-style) and paraphrase (PAWS-X-style).
    -   Place the data files in the `benchmark_data/` directory.
2.  **Implement Data Loader**: Add a loader function in `benchmark_loader/datautils.py` to load your new dataset.
3.  **Define Prompt Templates**: In `benchmark_loader/xglue_prompt_templates.py`, add prompt templates for your tasks, similar to the ones in `SELECTED_XGLUE_TASKS`.
4.  **Create Evaluation Spec**: In the `evaluation/` directory, create a new `EvalSpec` subclass for your language. You will need to implement the `extract_answer` method to parse the model's output.
5.  **Register Benchmark**: Add an entry for the new language in the `LINGUISTIC_BENCHMARKS` dictionary in `main.py`.
6.  **Update `apply_benchmark_dir`**: In `main.py`, update the `apply_benchmark_dir` function to correctly pass the data directory to your new loader and `EvalSpec`.
7.  **Add MMLU Data**: If you want to run `raw_eval` or `cross_eval`, add the corresponding MMLU test set for the new language in the `benchmark_data/mmlu` directory. The existing MMLU loader and evaluation logic should handle it automatically.
    -   Add language specific prompt templates in `benchmark_loader/mmlu_prompt_templates.py`.

## Benchmarks 📊
The project uses several benchmarks, some of which were pre-processed using scripts in the `dataset_utils/` directory.

*   **MMLU**: A subset of subjects, processed into a project-specific format.
*   **XGLUE**: Used for English, German, and French linguistic tasks.
*   **Italian**:
    *   Textual Entailment: [ELG Corpus 8121](https://live.european-language-grid.eu/catalogue/corpus/8121)
    *   Paraphrasing: [ZurichNLP/paws-x-italian](https://huggingface.co/datasets/ZurichNLP/paws-x-italian)
*   **Arabic**:
    *   Paraphrasing: [Arabic-Paraphrasing-Benchmark](https://github.com/marwah2001/Arabic-Paraphrasing-Benchmark)
*   **Hindi**:
    *   Paraphrasing: [ai4bharat/IndicParaphrase](https://huggingface.co/datasets/ai4bharat/IndicParaphrase), processed to add negative samples.

## Acknowledgements 🙏
This project builds upon the following key repositories:
*   **[Wanda](https://github.com/locuslab/wanda)**: The core pruning methodology. The implementation has been significantly adapted for our experiments, including optimizations for GPU memory, custom calibration data handling, and batch processing.
*   **[pruning on ToM](https://github.com/Itakello/prune_on_tom)**: Served as the initial project structure and early integration of Wanda, which have since been extensively modified.
Please note that, beyond the publicly available work in the original *pruning on ToM* repository, its author has not been directly involved in or contributed to the development of this project. The author’s name may appear in the contributors list because their commits are part of the inherited Git history from the original repository.

## License 📜
This project is licensed under the MIT License - see the `LICENSE` file for details.

