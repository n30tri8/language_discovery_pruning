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

    with torch.no_grad():
        calib_data = prepare_calibration(model, dataloader, max_cal_len, dev)

    # enable memory history, which will
    try:
        prune_wanda(model, calib_data, sparsity, device=dev)
    except torch.AcceleratorError as e:
        print(e)
    finally:
        print("saving log before exiting.")
        torch.cuda.memory._dump_snapshot("gpu_snapshot.pickle")
        exit(-1)

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         record_shapes=True,
    #         profile_memory=True
    # ) as prof:
    #     with record_function("prune_wanda"):
    #         try:
    #             prune_wanda(model, calib_data, sparsity, device=dev)
    #         except torch.AcceleratorError as e:
    #             print(e)
    #         finally:
    #             print("saving log before exiting.")
    #             prof.export_chrome_trace("trace.json")
    #             exit(-1)

    print("Wanda-based pruning done!")


def prepare_calibration(model, dataloader, max_len, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    # We'll gather the hidden input states (inps) for each calibration sample,
    # plus the attention_mask and position_ids (mirroring old logic).
    dtype = next(iter(model.parameters())).dtype
    nsamples = len(dataloader)
    inps = torch.zeros((nsamples, max_len, model.config.hidden_size), dtype=dtype)
    inps.requires_grad = False
    attention_masks = torch.zeros((nsamples, 1, max_len, max_len), dtype=torch.bool)
    position_embeddings_0 = torch.zeros((1, max_len, model.config.head_dim), dtype=torch.float16)
    position_embeddings_1 = torch.zeros((1, max_len, model.config.head_dim), dtype=torch.float16)
    position_embeddings_0.requires_grad = False
    position_embeddings_1.requires_grad = False
    cache = {'i': 0, 'position_embeddings_0': position_embeddings_0, 'position_embeddings_1': position_embeddings_1}

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
            idx = cache["i"]
            attn = kwargs.get("attention_mask")
            if attn is None:  # no padding
                token_mask = torch.ones(max_len, dtype=torch.bool)
                attn = token_mask[:max_len].unsqueeze(1) & token_mask[:max_len].unsqueeze(0)
                attn = attn.unsqueeze(0)
            elif attn.dim() == 4:
                attn = attn[0]
            else:
                raise RuntimeError(f"Unexpected attention_mask ndim={attn.dim()}")

            inps[idx, :max_len, :] = hidden_states
            attention_masks[idx, 0, :max_len, :max_len] = attn

            pe0, pe1 = kwargs.get("position_embeddings", None)
            if idx == 0:
                cache["position_embeddings_0"][0, :max_len, :] = pe0
                cache["position_embeddings_1"][0, :max_len, :] = pe1

            cache["i"] += 1
            raise ValueError  # early stop

    # Hook the first layer
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            _ = model(batch[0].to(dev), attention_mask=batch[1].to(dev), use_cache=False)
        except ValueError:
            pass
    # Restore the actual layer
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    outs.requires_grad = False
    calib_data = {
        "inps": inps,
        "attention_masks": attention_masks,
        "position_embeddings_0": cache["position_embeddings_0"],
        "position_embeddings_1": cache["position_embeddings_1"],
        "outs": outs,
    }
    model.config.use_cache = use_cache

    return calib_data
