import sys
from typing import Any, Dict
from luxai_s2.state import State
import numpy as np
import numpy.typing as npt
from gym import spaces


# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()


class SimpleUnitDiscreteController(Controller):

    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.unit_act_dims = self.no_op_dim_high

        self.robots_number = 10

        action_space = spaces.MultiDiscrete(
            [self.unit_act_dims for _ in range(self.robots_number)]
        )

        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def _map_action(self, id):
        action_queue = []
        if self._is_move_action(id):
            action_queue = [self._get_move_action(id)]
        elif self._is_transfer_action(id):
            action_queue = [self._get_transfer_action(id)]
        elif self._is_pickup_action(id):
            action_queue = [self._get_pickup_action(id)]
        elif self._is_dig_action(id):
            action_queue = [self._get_dig_action(id)]
        return action_queue

    def _create_unit_name(self, id):
        return f'unit_{id}'

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        shared_obs = obs["player_0"]
        lux_action = dict()
        units = shared_obs["units"][agent]
        obs = State.from_obs(obs[agent], self.env_cfg)

        for unit_id, unit_action_id in zip(units.keys(), action):
            # Note: map_action returns queue
            mapped_action = self._map_action(unit_action_id)
            if len(mapped_action) == 0 or self.is_action_valid(unit_id, unit_action_id, obs, mapped_action[0]):
                lux_action[unit_id] = mapped_action
        return lux_action

    def is_action_valid(self, unit_id, action, obs: State, mapped_action):
        if self._is_move_action(action):
            unit_position = obs.units["player_0"][unit_id].pos
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            move_dir = mapped_action[1]
            new_position = unit_position + move_deltas[move_dir]
            if not (new_position.x in range(obs.board.width)) or not (new_position.y in range(obs.board.height)):
                return False
        return True

    def split_logits_by_dim(self, logits):
        robot_id_logits = logits[:, :self.max_bots_count]
        unit_action_logits = logits[:, self.max_bots_count:
                                    self.max_bots_count + self.unit_act_dims]
        factory_action_logits = logits[:, -self.factory_act_count+1:]
        return robot_id_logits, unit_action_logits, factory_action_logits

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1: f_pos[0] + 2, f_pos[1] - 1: f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]

        unit_action_mask = np.zeros(self.unit_act_dims, dtype=bool)
        for unit_id in units.keys():

            # movement is always valid
            unit_action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0],
                                                      transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    unit_action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                unit_action_mask[
                    self.dig_dim_high - self.dig_act_dims: self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                unit_action_mask[
                    self.pickup_dim_high - self.pickup_act_dims: self.pickup_dim_high
                ] = True
                unit_action_mask[
                    self.dig_dim_high - self.dig_act_dims: self.dig_dim_high
                ] = False

            # no-op is always valid
            unit_action_mask[-1] = True
            break
        action_mask = np.ones((sum(self.action_space.nvec), ), dtype=bool)
        action_mask[self.max_bots_count:self.max_bots_count +
                    self.unit_act_dims] = unit_action_mask
        return action_mask
