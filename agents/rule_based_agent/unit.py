from agents.lux.kit import GameState


from abc import ABC, abstractmethod


class Unit(ABC):
    def __init__(self, player, key) -> None:
        self._key = key
        self._player = player

    @abstractmethod
    def get_game_unit(self, unit, obs: GameState):
        ...

    def act(self, obs: GameState):
        unit = self.get_game_unit(obs)
        objective = self.condition_objective(unit, obs)
        action = self._act(unit, objective, obs)
        return {
            self._key: action
        } if action is not None else None

    @abstractmethod
    def _act(self, unit, objective, obs: GameState):
        ...

    @abstractmethod
    def condition_objective(self, unit, obs: GameState):
        ...
