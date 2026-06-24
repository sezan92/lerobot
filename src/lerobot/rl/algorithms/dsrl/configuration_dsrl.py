from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class DSRLAlgorithmConfig(RLAlgorithmConfig):
    """DSRL algorithm configuration."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    grad_clip_norm: float = 1.0
    policy_update_freq: int = 1

    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_config: PreTrainedConfig) -> DSRLAlgorithmConfig:
        return cls(policy_config=policy_config)
