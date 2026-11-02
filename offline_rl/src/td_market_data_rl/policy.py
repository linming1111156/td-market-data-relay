from __future__ import annotations

from typing import Any

import torch


ARCHITECTURES = ("linear", "mlp_tanh8")


def build_actor(architecture: str, input_size: int,
                device: torch.device | str = "cpu") -> torch.nn.Module:
    if architecture == "linear":
        return torch.nn.Linear(input_size, 3, bias=True, device=device)
    if architecture == "mlp_tanh8":
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, 8, bias=True, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 3, bias=True, device=device),
        )
    raise ValueError(f"unsupported policy architecture: {architecture}")


def initialize_actor(actor: torch.nn.Module, architecture: str) -> None:
    if architecture == "linear":
        assert isinstance(actor, torch.nn.Linear)
        torch.nn.init.zeros_(actor.weight)
        torch.nn.init.zeros_(actor.bias)
        return
    if architecture == "mlp_tanh8":
        assert isinstance(actor, torch.nn.Sequential)
        torch.nn.init.xavier_uniform_(actor[0].weight)
        torch.nn.init.zeros_(actor[0].bias)
        torch.nn.init.zeros_(actor[2].weight)
        torch.nn.init.zeros_(actor[2].bias)
        return
    raise ValueError(f"unsupported policy architecture: {architecture}")


def serialize_actor(actor: torch.nn.Module, architecture: str) -> dict[str, Any]:
    if architecture == "linear":
        assert isinstance(actor, torch.nn.Linear)
        return {"weights": actor.weight.detach().cpu().tolist(),
                "bias": actor.bias.detach().cpu().tolist()}
    if architecture == "mlp_tanh8":
        assert isinstance(actor, torch.nn.Sequential)
        return {
            "hidden_weights": actor[0].weight.detach().cpu().tolist(),
            "hidden_bias": actor[0].bias.detach().cpu().tolist(),
            "output_weights": actor[2].weight.detach().cpu().tolist(),
            "output_bias": actor[2].bias.detach().cpu().tolist(),
        }
    raise ValueError(f"unsupported policy architecture: {architecture}")


def load_actor(model: dict[str, Any], input_size: int,
               device: torch.device | str = "cpu") -> torch.nn.Module:
    architecture = model.get("architecture", "linear")
    actor = build_actor(architecture, input_size, device)
    with torch.no_grad():
        if architecture == "linear":
            assert isinstance(actor, torch.nn.Linear)
            actor.weight.copy_(torch.tensor(model["weights"], dtype=torch.float32, device=device))
            actor.bias.copy_(torch.tensor(model["bias"], dtype=torch.float32, device=device))
        else:
            assert isinstance(actor, torch.nn.Sequential)
            actor[0].weight.copy_(torch.tensor(model["hidden_weights"], dtype=torch.float32, device=device))
            actor[0].bias.copy_(torch.tensor(model["hidden_bias"], dtype=torch.float32, device=device))
            actor[2].weight.copy_(torch.tensor(model["output_weights"], dtype=torch.float32, device=device))
            actor[2].bias.copy_(torch.tensor(model["output_bias"], dtype=torch.float32, device=device))
    return actor
