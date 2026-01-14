import torch
import torch.nn as nn
from tqdm import tqdm

from .layerwrapper import WrappedGPT


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration(model, dataloader):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dev = None
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    count_samples = len(dataloader)
    inps = [None] * count_samples
    attention_masks = [None] * count_samples
    outs = [None] * count_samples
    position_embeddings_0 = [None] * count_samples
    position_embeddings_1 = [None] * count_samples
    cache = {'i': 0}

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

            attn = kwargs.get("attention_mask")
            if attn is None:  # no padding, therefore we create a symmetric square for attending to full sequence
                token_mask = torch.ones(hidden_states.shape[1], dtype=torch.bool)
                attn = token_mask[:hidden_states.shape[1]].unsqueeze(1) & token_mask[:hidden_states.shape[1]].unsqueeze(0)
                attn = attn.unsqueeze(0)
            elif attn.dim() == 4:
                attn = attn[0]
            else:
                raise RuntimeError(f"Unexpected attention_mask ndim={attn.dim()}")
            attention_masks[idx] = attn.cpu()

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
        "attention_masks": attention_masks,
        "position_embeddings_0": position_embeddings_0,
        "position_embeddings_1": position_embeddings_1,
        "outs": outs
    }
    model.config.use_cache = use_cache

    return calib_data


def prune_wanda(model, calib_data, sparsity_ratio):
    """
    Modified Wanda function that uses already-collected calibration data
    (inps, attention_masks, position_embeddings) rather than reloading from dataset.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps = calib_data["inps"]
    outs = calib_data["outs"]
    attention_masks = calib_data["attention_masks"]
    position_embeddings_0 = calib_data["position_embeddings_0"]
    position_embeddings_1 = calib_data["position_embeddings_1"]
    count_samples = len(inps)

    layers = model.model.layers
    for i, layer in enumerate(tqdm(layers, desc="Processing layers")):
        subset = find_layers(layer)

        # For each sub-layer, wrap it so we can track input norms
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # Register hooks
        def make_hook(n):
            def f(_, x_in, x_out):
                # x_in is a tuple of (hidden_states,) for a typical forward
                # We'll pass both input & output to the wrapper
                wrapped_layers[n].add_batch(x_in[0].data, x_out.data)

            return f

        handles = []
        for name in wrapped_layers:
            h = subset[name].register_forward_hook(make_hook(name))
            handles.append(h)

        layer_dev = next(layer.parameters()).device
        # forward pass all calibration samples
        with torch.no_grad():
            for j in range(count_samples):
                outs[j] = layer(
                    inps[j].unsqueeze(0).to(layer_dev),
                    attention_mask=attention_masks[j].unsqueeze(0).to(layer_dev),
                    position_embeddings=(position_embeddings_0[j].to(layer_dev), position_embeddings_1[j].to(layer_dev))
                )[0]
        torch.cuda.empty_cache()

        # remove hooks
        for h in handles:
            h.remove()

        # Wanda: unstructured => pick top-K
        for name in tqdm(
                wrapped_layers.keys(),
                desc=f"Pruning sublayers in layer {i} name {name}",
                leave=False,
        ):
            # Weighted metric = abs(W) * sqrt( row-norm of input )
            W = subset[name].weight.data
            row_norms = torch.sqrt(wrapped_layers[name].scaler_row).reshape(1, -1)
            W_metric = torch.abs(W) * row_norms

            # unstructured pruning
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            # pick the fraction of smallest entries per-output
            k = int(W_metric.shape[1] * sparsity_ratio)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :k]
            W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero
        # Explicitly free memory
        del W_metric, W_mask, sort_res, indices, row_norms
        torch.cuda.empty_cache()

        # forward pass again so next layer sees the pruned representation
        with torch.no_grad():
            for j in range(count_samples):
                outs[j] = layer(
                    inps[j].unsqueeze(0).to(layer_dev),
                    attention_mask=attention_masks[j].unsqueeze(0).to(layer_dev),
                    position_embeddings=(position_embeddings_0[j].to(layer_dev), position_embeddings_1[j].to(layer_dev))
                )[0]

        # swap
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    del inps, outs, attention_masks, position_embeddings_0, position_embeddings_1
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
