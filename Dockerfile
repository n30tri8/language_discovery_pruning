# Stage 1: Prepare raw_model and install dependencies
FROM python:3.10-slim AS raw_model

# Set the working directory in the container
WORKDIR /app

COPY raw_model/ raw_model/

RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    unzip raw_model/models--meta-llama--Llama-3.1-8B-Instruct.zip -d raw_model/ && \
    rm raw_model/models--meta-llama--Llama-3.1-8B-Instruct.zip && \
    apt-get purge -y --auto-remove unzip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM raw_model AS app
# Copy the project files into the container
COPY analyze_pruning.py main.py mmlu_evaluation.py pruning.py utils.py benchmark_data submodules ./

# Command to run the application
ENTRYPOINT ["python", "main.py"]
CMD ["--test_num", "400", "--sparsity_ratios", "50"]
