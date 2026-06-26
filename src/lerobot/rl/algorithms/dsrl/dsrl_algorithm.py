from ..base import RLAlgorithm
from .configuratuin_dsrl import DSRLAlgorithmConfig
class DSRLAlgorithm(RLAlgorithm):
    config_class = DSRLAlgorithmConfig
    name="dsrl"
    def __init__(self, policy, DSRLPolicy, config: DSRLAlgorithmConfig):
        self.config = config
        self.policy = policy
        self.optimizer = dict[str, Optimizer] = {}
        ## TODO Sezan; complete