from agents.lux.kit import GameState, Factory
from enum import IntEnum, auto
from agents.rule_based_agent.unit import Unit


class FactoryObjective(IntEnum):
    GrowLichen = auto()
    BuildLightRobot = auto()
    BuildHeavyRobot = auto()
    CreateSupplyChain = auto()
    Wait = auto()


class UFactory(Unit):

    def __init__(self, player, key, supply_chain_lenghts: int, water_buffer: int, safe_watering_steps: int, max_light_robots: int) -> None:
        """
        :param supply_chain_lenghts: how many heavy robots should be in the supply chain
        :param water_buffer: how much water should be left in the factory after watering
        :param safe_watering_steps: how many steps the factory should be able to water existing lichen
        """
        super().__init__(player, key)
        self.supply_chain_lenghts = supply_chain_lenghts
        self.water_buffer = water_buffer
        self.safe_watering_steps = safe_watering_steps
        self.max_light_robots = max_light_robots

    def get_game_unit(self, obs: GameState):
        return obs.factories[self._player][self._key]

    def _act(self, factory: Factory, objective: FactoryObjective, obs: GameState):
        if objective == FactoryObjective.GrowLichen:
            return factory.water()
        if objective == FactoryObjective.BuildHeavyRobot:
            return factory.build_heavy()
        if objective == FactoryObjective.BuildLightRobot:
            return factory.build_light()

    def condition_objective(self, factory: Factory, obs: GameState):
        if factory.can_build_heavy(obs) and not self.supply_chain_exists(factory, obs):
            return FactoryObjective.BuildHeavyRobot
        if factory.can_build_light(obs) and self.supply_chain_exists(factory, obs):
            return FactoryObjective.BuildLightRobot
        if factory.cargo.water >= (factory.water_cost(obs) * self.safe_watering_steps + self.water_buffer):
            return FactoryObjective.GrowLichen

    def supply_chain_exists(self, factory: Factory, obs: GameState):
        # TODO very simplified at this moment
        heavy_count = len(
            [u for u in obs.units[self._player].values() if u.unit_type == "HEAVY"])
        return heavy_count >= self.supply_chain_lenghts
