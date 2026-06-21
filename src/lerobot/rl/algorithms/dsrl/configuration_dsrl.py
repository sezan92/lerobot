from __future__ import annotations

from dataclasses import dataclass,field

from lerobot.configs.policies import PreTrainedConfig
from ..configs import RLAlgorithmConfig
@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class DSRLAlgorithmConfig(RLAlgorithmConfig):
    """DSRL algorithm configuration."""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4