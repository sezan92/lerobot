from collections.abc import Iterator

import torch
from torch.optim import Optimizer

from lerobot.policies.dsrl.modeling_dsrl import DSRLPolicy
from lerobot.types import BatchType

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
            ## TODO : start from https://github.com/sezan92/RL_study/issues/65#issuecomment-5119276016