from __future__ import annotations

import torch


def integrated_gradients(
    model,
    inputs: torch.Tensor,
    scale: str,
    lead: int,
    baseline: torch.Tensor | None = None,
    steps: int = 64,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be >= 2")
    baseline = (
        torch.zeros_like(inputs) if baseline is None else baseline.to(inputs.device)
    )
    if baseline.shape != inputs.shape:
        baseline = baseline.expand_as(inputs)
    try:
        from captum.attr import IntegratedGradients

        def selected_output(value):
            prediction = model(value)[0][scale]
            return (
                prediction[..., 1][:, lead]
                if prediction.ndim == 3
                else prediction[:, lead]
            )

        model.eval()
        return IntegratedGradients(selected_output).attribute(
            inputs, baselines=baseline, n_steps=steps
        )
    except ImportError:
        pass
    delta = inputs - baseline
    grads = []
    model.eval()
    for alpha in torch.linspace(0, 1, steps, device=inputs.device):
        point = (baseline + alpha * delta).detach().requires_grad_(True)
        predictions, _ = model(point)
        value = predictions[scale]
        if value.ndim == 3:
            value = value[..., 1]  # median quantile
        score = value[:, lead].sum()
        grad = torch.autograd.grad(score, point, retain_graph=False)[0]
        grads.append(grad)
    stack = torch.stack(grads)
    average = (stack[0] + stack[-1] + 2 * stack[1:-1].sum(dim=0)) / (2 * (steps - 1))
    return delta * average


def batch_interpretability(
    model,
    inputs: torch.Tensor,
    feature_names: list[str],
    metadata: list[dict],
    scales=("spei_3", "spei_6", "spei_12"),
    lead: int = 0,
    steps: int = 64,
) -> dict:
    attributions = {s: [] for s in scales}
    for scale in scales:
        attr = integrated_gradients(model, inputs, scale, lead, steps=steps)
        attributions[scale] = attr.detach().cpu().numpy()
    model.eval()
    with torch.no_grad():
        _, aux = model(inputs)
    attn = aux["attention"].detach().cpu().numpy()
    if attn.ndim == 4:
        attn = attn.mean(axis=1)
    if attn.ndim == 3 and attn.shape[-1] == 1:
        attn = attn[..., 0]
    return {
        "feature_names": feature_names,
        "metadata": metadata,
        "integrated_gradients": attributions,
        "attention": attn,
        "warning": "Attributions and attention describe model behaviour and do not establish causal effects.",
    }
