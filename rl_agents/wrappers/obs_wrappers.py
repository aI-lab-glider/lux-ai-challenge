from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from luxai_s2.pyvisual.visualizer import Visualizer
from luxai_s2.state import State
from matplotlib import pyplot as plt


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    map_height = map_width = 200
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
        self.controller_robot_actions = 13
        self.queue_size = 20
        self.robot_count = 20
        self.observation_space = spaces.Box(0, 255, dtype=np.uint8, shape=(self.map_height, self.map_width, 3))
        # spaces.Dict({
        #     # 'map_state': spaces.Box(0, 255, dtype=np.uint8, shape=(self.map_height, self.map_width, 3)),
        #     # TODO: can it be dynamic??
        #     # 'robot_queues': spaces.Box(0, self.controller_robot_actions, dtype=np.uint8, shape=(self.robot_count, self.queue_size))
        # })
        # spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @classmethod
    def convert_obs(cls, obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = {}

        for agent in obs.keys():
            env_state = State.from_obs(obs[agent], env_cfg)
            vis = Visualizer(env_state)            
            vis.update_scene(env_state)
            img = vis._create_image_array(vis.surf, (cls.map_height, cls.map_width))
            observation[agent] = img
            # for unit in env_state.units[agent].values():
            #     unit.action_queue
        return observation
    