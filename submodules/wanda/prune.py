import copy
import gc

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

    layer_dev = next(model.model.layers[0].parameters()).device

    count_batches = len(dataloader)
    inps = [None] * count_batches
    attention_masks = [None] * count_batches
    position_embeddings = [None] * count_batches
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

            inps[idx] = hidden_states.cpu()
            inps[idx].requires_grad = False

            attn = kwargs.get("attention_mask")
            if attn is None:  # no padding, therefore we create a symmetric square for attending to full sequence
                token_mask = torch.ones(hidden_states.shape[1], dtype=torch.bool)
                attn = token_mask[:hidden_states.shape[1]].unsqueeze(1) & token_mask[:hidden_states.shape[1]].unsqueeze(
                    0)
                attn = attn.unsqueeze(0).unsqueeze(0)
            attention_masks[idx] = attn.cpu()

            pe0, pe1 = kwargs.get("position_embeddings", None)
            position_embeddings[idx] = (pe0.cpu(), pe1.cpu())

            cache["i"] += 1
            raise ValueError  # early stop

    # Hook the first layer
    layers = model.model.layers
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            batch = batch.to(layer_dev)
            _ = model(**batch, use_cache=False)
        except ValueError:
            pass
    # Restore the actual layer
    layers[0] = layers[0].module

    calib_data = {
        "inps": inps,
        "attention_masks": attention_masks,
        "position_embeddings": position_embeddings,
    }
    model.config.use_cache = use_cache

    return calib_data


def prune_wanda(model, calib_data, sparsity_ratio, complementary_pruning: bool = False):
    """
    Modified Wanda function that uses already-collected calibration data
    (inps, attention_masks, position_embeddings) rather than reloading from dataset.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Create copies of the lists to avoid modifying the original calib_data
    inps = copy.deepcopy(calib_data["inps"])
    attention_masks = calib_data["attention_masks"]
    position_embeddings = calib_data["position_embeddings"]
    count_batches = len(inps)
    outs = [None] * count_batches

    layers = model.model.layers
    for i, layer in enumerate(tqdm(layers, desc="Processing layers")):
        subset = find_layers(layer)

        # For each sub-layer, wrap it so we can track input norms in wrapped_layers[name].scaler_row
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # Register hooks
        def make_hook(n):
            def f(_, x_in, x_out):
                # x_in is a tuple of positional arguments given to the layer for a typical forward, x_in[0] is inps[j] here
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
            for j in range(count_batches):
                pe = (
                    position_embeddings[j][0].expand(inps[j].shape[0], -1, -1).to(layer_dev) if position_embeddings[j][
                                                                                                    0] is not None else None,
                    position_embeddings[j][1].expand(inps[j].shape[0], -1, -1).to(layer_dev) if position_embeddings[j][
                                                                                                    1] is not None else None
                )
                outs[j] = layer(
                    inps[j].to(layer_dev),
                    attention_mask=attention_masks[j].to(layer_dev),
                    position_embeddings=pe
                ).cpu()  # keep the outs on cpu since it is not needed for processing by gpu
        # free GPU memory
        del pe
        gc.collect()
        torch.cuda.empty_cache()

        # remove hooks
        for h in handles:
            h.remove()

        # Wanda: unstructured => pick top-K
        for name in tqdm(
                wrapped_layers.keys(),
                desc=f"Pruning sub-layers in layer {i} name {name}", leave=False,
        ):
            # Weighted metric = abs(W) * sqrt( row-norm of input )
            W = subset[name].weight.data
            row_norms = torch.sqrt(wrapped_layers[name].scaler_row).reshape(1, -1)
            W_metric = torch.abs(W) * row_norms
            del row_norms  # not need this anymore
            if complementary_pruning:
                sparsity_ratio = 1.0 - sparsity_ratio
            k = int(W_metric.shape[1] * sparsity_ratio)
            # pick the fraction of smallest OR complementary entries per-output
            pick_largest = True if complementary_pruning else False
            indices = torch.topk(W_metric, k, dim=-1, largest=pick_largest)[1]
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero
        # free GPU memory
        del W, W_metric, W_mask, indices
        # del wrapped_layers, contains wrapped_layers[sublayer].scaler_row per sublayer
        del wrapped_layers
        gc.collect()
        torch.cuda.empty_cache()

        # forward pass again so next layer sees the pruned representation
        with torch.no_grad():
            for j in range(count_batches):
                pe = (
                    position_embeddings[j][0].expand(inps[j].shape[0], -1, -1).to(layer_dev) if position_embeddings[j][
                                                                                                    0] is not None else None,
                    position_embeddings[j][1].expand(inps[j].shape[0], -1, -1).to(layer_dev) if position_embeddings[j][
                                                                                                    1] is not None else None
                )
                outs[j] = layer(
                    inps[j].to(layer_dev),
                    attention_mask=attention_masks[j].to(layer_dev),
                    position_embeddings=pe
                ).cpu()  # keep the outs on cpu since it is not needed for processing by gpu
        del pe

        # swap
        inps, outs = outs, inps
        # free GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    del inps, outs
    gc.collect()
    torch.cuda.empty_cache()
