"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

"""
Write an controller, that will decide about where to place factories
"""


# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
import os.path as osp
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory
from conv_rl_agent.wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(
            osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return self.bid_policy(step, obs, remainingOverageTime)

        game_state = obs_to_game_state(step, self.env_cfg, obs)

        if my_turn_to_place_factory(game_state.teams[self.player].place_first,
                                    step):
            return self.factory_placement_policy(step, obs, remainingOverageTime)
        return {}

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(
            zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area //
                                 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        spawn_loc = potential_spawns[np.random.randint(
            0, len(potential_spawns))]
        if not done_search:
            pos = spawn_loc

        metal = obs["teams"][self.player]["metal"]
        return dict(spawn=pos, metal=metal, water=metal)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(
            raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        if th.cuda.is_available():
            obs = obs.cuda()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(
                    self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )

            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features)
            # shape (1, N) where N=12 for the default controller
            logits = self.policy.policy.action_net(x)

            logits[~action_mask] = -1e8  # mask out invalid actions
            logits_by_dim = self.controller.split_logits_by_dim(logits)
            dists = [th.distributions.Categorical(
                logits=l) for l in logits_by_dim]
            action = [d.sample().cpu().numpy() for d in dists]
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, action
        )

        # commented code below adds watering lichen which can easily improve your agent
        # shared_obs = raw_obs[self.player]
        # factories = shared_obs["factories"][self.player]
        # for unit_id in factories.keys():
        #     factory = factories[unit_id]
        #     if 1000 - step < 50 and factory["cargo"]["water"] > 100:
        #         lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
