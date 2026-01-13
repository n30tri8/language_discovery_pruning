import torch
import torch.nn as nn

from submodules.wanda.prune import prune_wanda


@torch.no_grad()
def llama_sparsellm(model, dataloader, max_cal_len, sparsity) -> None:
    """
    Replacement of the old 'llama_sparsellm' that now calls Wanda's prune_wanda
    using the calibration data from 'dataloader'.
    """
    print("Starting Wanda-based pruning...")

    with torch.no_grad():
        calib_data = prepare_calibration(model, dataloader, max_cal_len)

    prune_wanda(model, calib_data, sparsity)

    print("Wanda-based pruning done!")


def prepare_calibration(model, dataloader, max_len):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dev = None
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    count_samples = len(dataloader)
    inps = [None] * count_samples
    outs = [None] * count_samples
    position_embeddings_0 = [None] * count_samples
    position_embeddings_1 = [None] * count_samples
    cache = {'i': 0}

    # # We'll gather the hidden input states (inps) for each calibration sample,
    # # plus the attention_mask and position_ids (mirroring old logic).
    # dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((count_samples, max_len, model.config.hidden_size), dtype=dtype)
    # inps.requires_grad = False
    # attention_masks = torch.zeros((count_samples, 1, max_len, max_len), dtype=torch.bool)
    # position_embeddings_0 = torch.zeros((1, max_len, model.config.head_dim), dtype=torch.float16, device="cpu")
    # position_embeddings_1 = torch.zeros((1, max_len, model.config.head_dim), dtype=torch.float16, device="cpu")
    # position_embeddings_0.requires_grad = False
    # position_embeddings_1.requires_grad = False
    # cache = {'i': 0, 'position_embeddings_0': position_embeddings_0, 'position_embeddings_1': position_embeddings_1}

    # We'll use a forward hook on the first layer to capture the hidden states
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # Optionally copy known attributes Qwen2 expects
            for attr in ("attention_type", "config"):
                if hasattr(module, attr):
                    setattr(self, attr, getattr(module, attr))

        def __getattr__(self, name):
            # Delegate missing attributes to the wrapped module
            if name != "module" and hasattr(self.__dict__.get("module", None), name):
                return getattr(self.__dict__["module"], name)
            return super().__getattr__(name)

        def forward(self, hidden_states, **kwargs):
            idx: int = cache["i"]

            inps[idx] = hidden_states[0].cpu()
            inps[idx].requires_grad = False

            pe0, pe1 = kwargs.get("position_embeddings", None)
            position_embeddings_0[idx] = pe0.cpu()
            position_embeddings_1[idx] = pe1.cpu()

            cache["i"] += 1
            raise ValueError  # early stop

    # Hook the first layer
    layers[0] = Catcher(layers[0])
    for sample in dataloader:
        try:
            _ = model(sample.to(dev), use_cache=False)
        except ValueError:
            pass
    # Restore the actual layer
    layers[0] = layers[0].module

    calib_data = {
        "inps": inps,
        "position_embeddings_0": position_embeddings_0,
        "position_embeddings_1": position_embeddings_1,
        "outs": outs
    }
    model.config.use_cache = use_cache

    return calib_data
