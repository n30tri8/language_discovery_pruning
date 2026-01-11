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


def prune_wanda(model, calib_data, sparsity_ratio, device):
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
    nsamples = inps.shape[0]

    layers = model.model.layers
    for i, layer in enumerate(tqdm(layers, desc="Processing layers")):
        subset = find_layers(layer)

        # Determine the device for the current layer and move tensors
        layer_dev = next(layer.parameters()).device
        inps = inps.to(layer_dev)
        outs = outs.to(layer_dev)
        attention_masks = attention_masks.to(layer_dev)
        position_embeddings_0 = position_embeddings_0.to(layer_dev)
        position_embeddings_1 = position_embeddings_1.to(layer_dev)
        torch.cuda.empty_cache()

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

        # forward pass all calibration samples
        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_masks[j].unsqueeze(0),
                    position_embeddings=(position_embeddings_0, position_embeddings_1)
                )[0]

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
            for j in range(nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_masks[j].unsqueeze(0),
                    position_embeddings=(position_embeddings_0, position_embeddings_1)
                )[0]

        # swap
        inps, outs = outs, inps
        # torch.cuda.empty_cache()

    del inps, outs, attention_masks, position_embeddings_0, position_embeddings_1
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
