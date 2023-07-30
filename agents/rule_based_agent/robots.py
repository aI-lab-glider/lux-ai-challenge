from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from agents.lux.kit import GameState, Unit as GUnit, Board
from enum import IntEnum, auto
from agents.rule_based_agent.unit import Unit
from queue import PriorityQueue


class ResourceType(IntEnum):
    Ore = auto()
    Ice = auto()


@dataclass(order=True)
class step:
    distance_to_goal: int
    position: np.ndarray = field(compare=False)
    direction: int = field(compare=False)


def manhattan(pos1, pos2):
    diff = pos1 - pos2
    return sum(abs(diff))


def astar(from_: np.ndarray, to_: np.ndarray, board: Board, visited: list = None):
    if visited is None:
        visited = []
    if manhattan(from_, to_) == 0:
        return [step(0, to_, 0)]

    #    direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    move_deltas = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    queue = PriorityQueue[step]()
    for direction, delta in enumerate(move_deltas, 1):
        new_pos = from_ + delta
        if 0 < new_pos[0] < board.ice.shape[0] and 0 < new_pos[1] < board.ice.shape[1] and tuple(new_pos) not in visited:
            queue.put(step(manhattan(new_pos, to_), new_pos, direction))
    while not queue.empty():
        candidate = queue.get()
        visited.append(tuple(candidate.position))
        path_from_candidate = astar(
            candidate.position, to_, board, visited)
        if path_from_candidate is not None:
            return [candidate, *path_from_candidate]


class LightUnitObjective(IntEnum):
    CollectIce = auto()
    CollectOre = auto()
    TransferResource = auto()
    Recharge = auto()


class LightRobot(Unit):
    def __init__(self, player, key, operating_power_level, max_cargo: int, priority_resource: ResourceType, min_allowed_power: int) -> None:
        super().__init__(player, key)
        self.operating_power_level = operating_power_level
        self.max_cargo = max_cargo
        self.priority_resource = priority_resource
        self.min_allowed_power = min_allowed_power

    def get_game_unit(self, obs: GameState):
        return obs.units[self._player][self._key]

    def _act(self, unit: GUnit, objective: LightUnitObjective, obs: GameState):
        action = self._select_action(unit, objective, obs)
        return [action]

    def _select_action(self, unit: GUnit, objective: LightUnitObjective, obs: GameState):
        if objective == LightUnitObjective.CollectIce:
            distance, closest_ice = self._find_closest_ice(unit, obs)
            if distance > 1:
                next_step = self._find_next_step(unit, closest_ice, obs)
                return unit.move(next_step)
            else:
                return unit.dig()
        if objective == LightUnitObjective.CollectOre:
            distance, closest_ore = self._find_closest_ore(unit, obs)
            if distance > 1:
                next_step = self._find_next_step(unit, closest_ore, obs)
                return unit.move(next_step)
            else:
                return unit.dig()
        if objective == LightUnitObjective.TransferResource:
            distance, closest_drop_point = self._find_closest_drop_point(
                unit, obs)
            next_step = self._find_next_step(unit, closest_drop_point, obs)
            if distance > 1:
                return unit.move(next_step)
            else:
                return unit.transfer(next_step)
        if objective == LightUnitObjective.Recharge:
            return unit.recharge(self.operating_power_level)

    def _find_closest_ice(self, unit: GUnit, obs: GameState):
        board = obs.board
        return min([
            (manhattan(unit.pos, ice), ice) for ice in zip(*board.ice.nonzero())
        ], key=lambda a: a[0])

    def _find_closest_ore(self, unit: GUnit, obs: GameState):
        board = obs.board
        return min([
            (manhattan(unit.pos, ore), ore) for ore in zip(*board.ore.nonzero())
        ], key=lambda a: a[0])

    def _find_closest_drop_point(self, unit: GUnit, obs: GameState):
        # TODO: take supply chains into account
        board = obs.board
        return min([
            (manhattan(unit.pos, factory), factory) for factory in zip(*board.factory_occupancy_map.nonzero())
        ], key=lambda a: a[0])

    def _find_next_step(self, unit: GUnit, destination: np.ndarray, obs: GameState):
        path = astar(unit.pos, destination, obs.board)
        return path[0].direction

    def condition_objective(self, unit: GUnit, obs: GameState):
        if unit.cargo.ice >= self.max_cargo or unit.cargo.metal >= self.max_cargo:
            return LightUnitObjective.TransferResource
        # TODO: how to know if I should collect ice or ore?
        if unit.power < self.min_allowed_power and self.min_allowed_power is not None:
            return LightUnitObjective.Recharge
        if self.priority_resource == ResourceType.Ice:
            return LightUnitObjective.CollectIce
        if self.priority_resource == ResourceType.Ore:
            return LightUnitObjective.CollectOre
