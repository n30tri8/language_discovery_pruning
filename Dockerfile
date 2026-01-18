# Stage 1: Prepare raw_model and install dependencies
FROM python:3.10-slim AS raw_model

# Set the working directory in the container
WORKDIR /app

COPY raw_model/ raw_model/

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM lang_discovery/raw_model/qwen2.5_7b AS app
# Copy the project files into the container
COPY benchmark_data ./benchmark_data
COPY submodules ./submodules
COPY benchmark_loader ./benchmark_loader
COPY evaluation ./evaluation
COPY main.py utils.py ./

# Command to run the application
ENTRYPOINT ["python", "main.py"]
CMD ["--model", "Qwen/Qwen2.5-7B-Instruct", "--test_num", "300", "--sparsity_ratios", "50", "--run", "raw_eval", "prune", "cross_eval", "--languages", "en"]
