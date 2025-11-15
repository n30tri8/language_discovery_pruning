import torch
import torch.nn as nn
from tqdm import tqdm

from .layerwrapper import WrappedGPT


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prune_wanda(model, calib_data, args, device=torch.device("cuda:0")):
    """
    Modified Wanda function that uses already-collected calibration data
    (inps, attention_masks, position_embeddings) rather than reloading from dataset.
    """

    inps = calib_data["inps"]
    attention_masks = calib_data["attention_masks"]
    position_embeddings = calib_data["position_embeddings"]

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)

    nsamples = inps.shape[0]

    # We'll create an 'outs' buffer for the next layer's input
    outs = torch.zeros_like(inps)

    for i, layer in enumerate(tqdm(layers, desc="Processing layers")):
        # Move layer to device
        layer = layer.to(device)
        subset = find_layers(layer)

        # For each sub-layer, wrap it so we can track input norms
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # Register hooks
        handles = []

        def make_hook(n):
            def f(_, x_in, x_out):
                # x_in is a tuple of (hidden_states,) for a typical forward
                # We'll pass both input & output to the wrapper
                wrapped_layers[n].add_batch(x_in[0].data, x_out.data)

            return f

        for n in wrapped_layers:
            h = subset[n].register_forward_hook(make_hook(n))
            handles.append(h)

        # forward pass all calibration samples
        for j in range(nsamples):
            with torch.no_grad():

                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_masks[j],
                    position_embeddings=position_embeddings[j],
                )[0]

        # remove hooks
        for h in handles:
            h.remove()

        # Wanda: unstructured => pick top-K or do N:M if needed
        for n in tqdm(
            wrapped_layers.keys(),
            desc=f"Processing sublayers in layer {i}",
            leave=False,
        ):
            # Weighted metric = abs(W) * sqrt( row-norm of input )
            W = subset[n].weight.data
            row_norms = torch.sqrt(wrapped_layers[n].scaler_row).reshape(1, -1)
            W_metric = torch.abs(W) * row_norms

            if args.sparsity_type == "unstructured":
                # unstructured
                mask = torch.zeros_like(W_metric, dtype=torch.bool)
                # fraction = args.sparsity_ratio
                fraction = args.sparsity_ratio  # e.g. 0.5
                # pick the fraction of smallest entries in W_metric
                k = int(W_metric.numel() * fraction)
                if k < 1:
                    continue
                # flatten & topk
                # We want the smallest k. topk(..., largest=False).
                threshold = torch.topk(W_metric.flatten(), k, largest=False)[0][-1]
                mask = W_metric <= threshold
                # zero them out
                W[mask] = 0.0
            else:
                # for other patterns, N:M can be adapted similarly
                pass

        # forward pass again so next layer sees the pruned representation
        for j in range(nsamples):
            with torch.no_grad():

                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_masks[j],
                    position_embeddings=position_embeddings[j],
                )[0]

        # swap
        layers[i] = layer.cpu()
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
