import torch
import torch.nn as nn

from submodules.wanda.prune import prune_wanda


@torch.no_grad()
def llama_sparsellm(model, dataloader, max_cal_len, dev, sparsity) -> None:
    """
    Replacement of the old 'llama_sparsellm' that now calls Wanda's prune_wanda
    using the calibration data from 'dataloader'.
    """
    print("Starting Wanda-based pruning...")

    use_cache = model.config.use_cache

    with torch.no_grad():
        calib_data = prepare_calibration(model, dataloader, max_cal_len, dev)

    prune_wanda(model, calib_data, sparsity, device=dev)

    model.config.use_cache = use_cache

    print("Wanda-based pruning done!")


def prepare_calibration(model, dataloader, max_len, dev):
    model.config.use_cache = False
    layers = model.model.layers
    # We'll gather the hidden input states (inps) for each calibration sample,
    # plus the attention_mask and position_ids (mirroring old logic).
    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    inps = torch.zeros(
        (nsamples, max_len, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps.requires_grad = False
    attention_masks = [None] * nsamples
    # We'll use a forward hook on the first layer to capture the hidden states
    cache = {'i': 0, 'position_ids': None}

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
            idx = cache["i"]
            inps[idx, : hidden_states.size(1)] = hidden_states
            attention_masks[idx] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache["i"] += 1
            raise ValueError  # early stop

    # Hook the first layer
    layers[0] = Catcher(layers[0]).to(dev)
    for batch in dataloader:
        try:
            inp_ids = batch[0]
            _ = model(inp_ids, attention_mask=batch[1], use_cache=False)
        except ValueError:
            pass
    # Restore the actual layer
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    calib_data = {
        "inps": inps,
        "attention_masks": attention_masks,
        "position_ids": cache['position_ids'],
        "outs": outs
    }

    return calib_data
