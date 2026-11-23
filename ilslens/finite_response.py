"""Predict a finite source edit from clean donor and recipient decoder trajectories."""

import torch

from .hooks import tensor_output


def partial_response(replay, state, source_layer, landmark, start):
    """Propagate the edit only to the landmark; return changes at readout positions."""
    if not 0 <= source_layer < landmark < len(replay.states) - 1:
        raise ValueError("The landmark must be between the source and final blocks")
    for i in range(source_layer + 1, landmark + 1):
        state = tensor_output(replay.model.model.layers[i](state, **replay.kwargs[i]))
    return state[0, start:] - replay.states[landmark][0, start:]


def secant_transport(partial, landmark_difference, endpoint_difference, orthogonal_gain):
    """Transport the donor-aligned component and rescale the remaining displacement."""
    squared_length = landmark_difference.square().sum(dim=-1, keepdim=True).clamp_min(1e-18)
    projection = (partial * landmark_difference).sum(dim=-1, keepdim=True) / squared_length
    scale = (endpoint_difference.square().sum(dim=-1, keepdim=True) / squared_length).sqrt()
    # With no measurable donor direction, fall back to unscaled residual continuation.
    scale = torch.where(
        landmark_difference.square().sum(dim=-1, keepdim=True) > 1e-18, scale, torch.ones_like(scale)
    )
    return projection * endpoint_difference + orthogonal_gain * scale * (
        partial - projection * landmark_difference
    )


def prefix_secants(partial, landmarks, endpoints, start):
    """Use past clean-token secants to interpolate the remaining response.

    Return the constant and gain-dependent terms of the constrained estimator,
    plus a ridge-only comparator. Each output uses only its causal prefix.
    """
    offsets, gains, ridge = [], [], []
    for j, patch in enumerate(partial):
        i = start + j
        z, y = landmarks[: i + 1], endpoints[: i + 1]
        length = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8)
        x, out = z / length, y / length
        ident = torch.eye(len(x), device=x.device, dtype=x.dtype)
        ridge.append(torch.linalg.solve(x @ x.T + ident, x @ patch) @ out)

        anchor = z[-1]
        norm_sq = anchor.dot(anchor).clamp_min(1e-18)
        along = (x @ anchor) / norm_sq
        cross = x - along[:, None] * anchor
        weight = torch.linalg.solve(cross @ cross.T + ident, cross @ patch)
        a = patch.dot(anchor) / norm_sq
        scale = torch.where(
            anchor.dot(anchor) > 1e-18, torch.linalg.vector_norm(y[-1]) / norm_sq.sqrt(), 1.0
        )
        offsets.append(a * y[-1] + weight @ (out - along[:, None] * y[-1]))
        gains.append(scale * (patch - a * anchor - weight @ cross))
    return torch.stack(offsets), torch.stack(gains), torch.stack(ridge)
