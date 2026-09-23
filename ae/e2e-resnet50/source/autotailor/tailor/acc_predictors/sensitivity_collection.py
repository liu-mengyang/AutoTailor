"""Image evaluation primitives used by the training-only sensitivity collector."""

import copy


def calibrate_batch_norm(model, loader, device):
    """OFA-style sample-weighted batch moments on the calibration loader only."""
    import torch
    import torch.nn.functional as functional

    working = copy.deepcopy(model).to(device).eval()
    moments = {}
    for name, module in working.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or module.running_mean is None:
            continue
        moments[name] = {"n": 0, "mean": 0, "var": 0}

        def make_forward(bn, accumulator):
            def forward(x):
                dimensions = [0, *range(2, x.ndim)]
                mean = x.mean(dimensions)
                shape = [1, -1, *([1] * (x.ndim - 2))]
                variance = (x - mean.view(shape)).square().mean(dimensions)
                accumulator["n"] += x.size(0)
                accumulator["mean"] += mean.detach() * x.size(0)
                accumulator["var"] += variance.detach() * x.size(0)
                return functional.batch_norm(x, mean, variance, bn.weight, bn.bias, False, 0.0, bn.eps)
            return forward

        module.forward = make_forward(module, moments[name])
    if not moments:
        return 0
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            working(images.to(device))
            seen += images.size(0)
        for name, module in model.named_modules():
            if name in moments:
                accumulator = moments[name]
                if not accumulator["n"]:
                    # A rebuilt subnet can retain modules outside its active graph.
                    continue
                module.running_mean.copy_(accumulator["mean"] / accumulator["n"])
                module.running_var.copy_(accumulator["var"] / accumulator["n"])
    return seen


def evaluate_top1(model, loader, device):
    """Count actual images, including a short last batch; never recalibrate BN."""
    import torch

    correct = count = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            prediction = model(images.to(device)).argmax(1).cpu()
            correct += int((prediction == labels.cpu()).sum().item())
            count += labels.numel()
    if not count:
        raise ValueError("Accuracy image loader is empty")
    return {"correct": correct, "image_count": count, "accuracy": 100.0 * correct / count}
