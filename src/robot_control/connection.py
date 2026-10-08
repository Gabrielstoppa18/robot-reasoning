"""Conexao com uma instancia do CoppeliaSim aberta (ZMQ remote API)."""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class SimConnection:
    """Wrapper fino sobre o RemoteAPIClient: expoe `sim` e `simIK` prontos."""

    def __init__(self, host: str = "localhost", port: int = 23000):
        self.client = RemoteAPIClient(host, port)
        self.sim = self.client.require("sim")
        self.simIK = self.client.require("simIK")

    def start_simulation(self):
        self.sim.startSimulation()

    def stop_simulation(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            pass

    def step(self):
        self.sim.step()

    def set_stepping(self, enabled: bool):
        self.sim.setStepping(enabled)
