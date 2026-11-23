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
    scale = torch.where(landmark_difference.square().sum(dim=-1, keepdim=True) > 1e-18,
                        scale, torch.ones_like(scale))
    return projection * endpoint_difference + orthogonal_gain * scale * (partial - projection * landmark_difference)
