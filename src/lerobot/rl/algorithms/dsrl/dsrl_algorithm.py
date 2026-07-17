import torch

from ..base import RLAlgorithm
from .configuratuin_dsrl import DSRLAlgorithmConfig


class DSRLAlgorithm(RLAlgorithm):
    config_class = DSRLAlgorithmConfig
    name = "dsrl"

    def __init__(self, policy: DSRLPolicy, config: DSRLAlgorithmConfig):
        self.config = config
        self.policy = policy
        self.optimizer = dict[str, Optimizer] = {}
        self._optimization_step: int = 0

    def make_optimizers_and_scheduler(self):
        cfg = self.config
        self.optimizers = {
            "critic_action": torch.optim.Adam(
                self.policy.action_critic_ensamble.parameters(), lr=cfg.critic_lr
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
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.policy.action_critic_ensemble.parameters()).item()
        self.optimizers["critic_action"].step()
        stats = TrainingStats(
            losses={"loss_critic": loss_critic.item()},
            grad_norms={"critic_grad": critic_grad},
        )
        # Update 2026/07/17 

    # TODO: sezan: complete and understand what you are doing.
