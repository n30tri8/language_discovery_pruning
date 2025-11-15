import random

from datasets import load_dataset


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids) -> None:
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(seqlen, tokenizer):
    # Load test dataset
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    # Tokenize text with truncation using seqlen
    testenc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen)
    return testenc
