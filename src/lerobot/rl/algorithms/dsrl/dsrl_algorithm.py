from collections.abc import Iterator
from typing import Any

import torch
from torch.optim import Optimizer

from lerobot.policies.dsrl.modeling_dsrl import DSRLPolicy
from lerobot.types import BatchType
from lerobot.utils.transition import move_state_dict_to_device

from ..base import RLAlgorithm
from ..configs import TrainingStats
from .configuration_dsrl import DSRLAlgorithmConfig


class DSRLAlgorithm(RLAlgorithm):
    config_class = DSRLAlgorithmConfig
    name = "dsrl"

    def __init__(self, policy: DSRLPolicy, config: DSRLAlgorithmConfig):
        self.config = config
        self.policy_config = config.policy_config
        self.policy = policy
        self.optimizers = dict[str, Optimizer] = {}
        self._optimization_step: int = 0
        self._move_to_device()

    def make_optimizers_and_scheduler(self):
        cfg = self.config
        self.optimizers = {
            "critic_action": torch.optim.Adam(
                self.policy.action_critic_ensemble.parameters(), lr=cfg.critic_lr
            ),
            "critic_noise": torch.optim.Adam(
                self.policy.noise_critic.parameters(),
                lr=cfg.critic_lr,
            ),
            "noise_actor": torch.optim.Adam(
                self.policy.noise_actor.parameters(),
                lr=cfg.critc_lr,
            ),
            "temperature": torch.optim.Adam([self.policy.log_alpha], lr=cfg.temperature_lr),
        }

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        clip = self.config.grad_clip_norm  # sezan: why?
        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch)

        # Phase 1: Actor Critic Update (TD-learning)
        loss_dict = self.policy.forward(fb, model="critic_action")
        loss_critic = loss_dict["loss_critic"]
        self.optimizers["critic_action"].zero_grad()
        loss_critic.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.policy.action_critic_ensemble.parameters()).item()
        self.optimizers["critic_action"].step()
        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic_grad": critic_grad},
        )
        loss_dict = self.policy.forward(fb, model="critic_noise")
        loss_noise_critic = loss_dict["loss_critic_noise"]
        self.optimizers["critic_noise"].zero_grad()
        loss_noise_critic.backward()
        noise_critic_grad = torch.nn.utils.clip_grad_norm_(
            self.policy.noise_critic.parameters(), max_norm=clip
        ).item()
        stats.losses["loss_noise_critic"] = loss_noise_critic.item()
        stats.grad_norms["critic_noise"] = noise_critic_grad

        if self._optimization_step % self.config.policy_update_freq == 0:
            loss_dict = self.policy.forward(fb, model="noise_actor")
            loss_noise_actor = loss_dict["loss_noise_actor"]
            self.optimizers["noise_actor"].zero_grad()
            loss_noise_actor.backward()
            noise_actor_grad = torch.nn.utils.clip_grad_norm_(
                self.policy.noise_actor.parameters(), max_norm=clip
            ).item()
            self.optimizers["noise_actor"].step()

            loss_dict = self.policy.forward(fb, model="temperature")
            loss_temperature = loss_dict["loss_temperature"]
            self.optimizers["temperature"].zero_grad()
            loss_temperature.backward()
            temp_grad = torch.nn.utils.clip_grad_norm_([self.policy.log_alpha], max_norm=clip)
            self.optimizers["temperature"].step()

            stats.losses["loss_noise_actor"] = loss_noise_actor.item()
            stats.losses["loss_temperature"] = loss_temperature.item()
            stats.grad_norms["noise_actor"] = noise_actor_grad
            stats.grad_norms["temperature"] = temp_grad
            stats.extra["temperature"] = self.policy.temperature

        self.policy.update_target_networks()
        self._optimization_step += 1

        return stats

    def get_weights(self) -> dict[str, Any]:
        """Send noise_actor state dict to actors."""
        return {"noise_actor": move_state_dict_to_device(self.policy.noise_actor.state_dict(), device="cpu")}

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load noise_actor weights from learner."""
        noise_actor_sd = move_state_dict_to_device(weights["noise_actor"], device)
        self.policy.noise_actor.load_state_dict(noise_actor_sd)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Algorithm-owned trainable tensors (critics, targets)."""
        bundle: dict[str, torch.Tensor] = {}
        for k, v in self.policy.action_critic_ensemble.state_dict().items():
            bundle[f"action_critic_ensemble.{k}"] = v
        for k, v in self.policy.action_critic_target.state_dict().items():
            bundle[f"action_critic_target.{k}"] = v
        for k, v in self.policy.noise_critic.state_dict().items():
            bundle[f"noise_critic.{k}"] = v
        return bundle

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], device: str | torch.device = "cpu"):
        """Load the model from a given state_dict dictionary."""
        raise NotImplementedError
        # TODO: 2026/09/07 start from "load_state_dict"
