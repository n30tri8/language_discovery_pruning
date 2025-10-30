from submodules.SparseLLM.datautils import get_mmlu

def prepare_calibration(tokenizer, benchmark_data_dir, subject, lang, train_num, seed):
    """Prepare calibration data for pruning."""
    trainloader, _ = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=train_num, test_num=0)
    max_cal_len = max(inp.shape[1] for inp, _, _ in trainloader)
    return trainloader, max_cal_len


