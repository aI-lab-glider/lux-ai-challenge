import random
from agents.lux.kit import obs_to_game_state, GameState
from agents.lux.config import EnvConfig
from agents.lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

from agents.rule_based_agent.factory import UFactory
from agents.rule_based_agent.robots import LightRobot, ResourceType


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = potential_spawns[np.random.randint(
                    0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from agents.lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        actions = {}
        for factory in game_state.factories[self.player]:
            unit = UFactory(self.player, factory, 0, 0, 0, 3)
            action = unit.act(game_state)
            if action is not None:
                actions = {**actions, **action}

        for unit in game_state.units[self.player]:
            unit = LightRobot(self.player, unit, 10, 50,
                              random.choice([ResourceType.Ice, ResourceType.Ore]), 10)
            action = unit.act(game_state)
            if action is not None:
                actions = {**actions, **action}
        return actions
