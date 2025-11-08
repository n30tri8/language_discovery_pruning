# Stage 1: Prepare raw_model and install dependencies
FROM python:3.10-slim AS raw_model

# Set the working directory in the container
WORKDIR /app

COPY raw_model/ raw_model/

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM lang_discovery/raw_model/llama3.1_8b AS app
# Copy the project files into the container
COPY benchmark_data ./benchmark_data
COPY submodules ./submodules
COPY main.py mmlu_evaluation.py pruning.py utils.py ./

# Command to run the application
ENTRYPOINT ["python", "main.py"]
CMD ["--test_num", "400", "--sparsity_ratios", "50"]
