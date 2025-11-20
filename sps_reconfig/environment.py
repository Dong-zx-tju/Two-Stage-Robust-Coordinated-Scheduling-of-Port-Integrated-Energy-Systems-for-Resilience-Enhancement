"""
Environment describing a simplified medium-voltage DC shipboard power system.

The implementation is intentionally lightweight so that the DQN-MASK training
loop can run in a CPU-only environment while preserving the main concepts
from the paper, including weighted loads, switchable lines, and action masks
that rule out unsafe operations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .graph import Edge, Node, ShipPowerGraph


@dataclass
class EnvConfig:
    horizon: int = 30
    base_reward_scale: float = 1.0
    overload_penalty: float = 3.0
    island_penalty: float = 5.0
    switch_cost: float = 0.01
    random_faults: int = 2
    seed: int = 0


@dataclass
class PowerState:
    switches: np.ndarray
    supplied: np.ndarray
    step_count: int = 0
    faulted_edges: List[int] = field(default_factory=list)


class ShipPowerEnv:
    """Small RL-friendly environment for shipboard distribution reconfiguration."""

    def __init__(self, graph: ShipPowerGraph, config: EnvConfig):
        self.graph = graph
        self.cfg = config
        self.rng = random.Random(self.cfg.seed)
        # Each switchable edge maps to an action; we also add a no-op action to
        # avoid states where every toggle would violate safety rules.
        self.action_size = len([e for e in self.graph.edges if e.switchable]) + 1
        self.reset()

    def reset(self) -> np.ndarray:
        """Applies random faults and returns the initial observation."""
        # all normally closed lines are energized
        switches = np.array([1 if e.normally_closed else 0 for e in self.graph.edges], dtype=np.float32)

        # apply stochastic faults to non-generator edges
        candidate_edges = [i for i, edge in enumerate(self.graph.edges) if edge.switchable]
        self.rng.shuffle(candidate_edges)
        faulted = candidate_edges[: self.cfg.random_faults]
        for idx in faulted:
            switches[idx] = 0.0
        supplied = self._compute_supplied_power(switches)
        self.state = PowerState(switches=switches, supplied=supplied, step_count=0, faulted_edges=faulted)
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action {action}; must be in [0, {self.action_size})")

        switch_idx = self._action_to_edge(action)
        if switch_idx is not None:  # None corresponds to a no-op
            self.state.switches[switch_idx] = 1 - self.state.switches[switch_idx]
            self.state.supplied = self._compute_supplied_power(self.state.switches)
        self.state.step_count += 1

        reward = self._reward()
        done = self.state.step_count >= self.cfg.horizon
        info = {"faulted_edges": self.state.faulted_edges}
        return self._observation(), reward, done, info

    def _observation(self) -> np.ndarray:
        load_status = self.state.supplied.astype(np.float32)
        switch_state = self.state.switches.astype(np.float32)
        return np.concatenate([load_status, switch_state], dtype=np.float32)

    def _action_to_edge(self, action: int) -> int:
        switchable_indices = [i for i, e in enumerate(self.graph.edges) if e.switchable]
        if action == self.action_size - 1:  # dedicated no-op action
            return None
        return switchable_indices[action]

    def legal_actions(self) -> np.ndarray:
        """Returns a binary mask of actions that keep the system connected and under capacity."""
        mask = np.ones(self.action_size, dtype=np.float32)
        for action in range(self.action_size - 1):
            edge_idx = self._action_to_edge(action)
            toggled = self.state.switches.copy()
            toggled[edge_idx] = 1 - toggled[edge_idx]
            supplied = self._compute_supplied_power(toggled)
            if self._violates_connectivity(toggled) or self._overloaded(toggled, supplied):
                mask[action] = 0.0
        mask[-1] = 1.0  # always allow no-op
        return mask

    # --- physics helpers -------------------------------------------------
    def _compute_supplied_power(self, switches: np.ndarray) -> np.ndarray:
        """Very small approximation of power distribution.

        A generator covers loads reachable through closed edges without
        exceeding line capacity. Loads are satisfied proportionally when
        multiple generators can reach the same bus.
        """
        n = len(self.graph.nodes)
        adjacency = {i: [] for i in range(n)}
        for idx, edge in enumerate(self.graph.edges):
            if switches[idx] > 0.5:  # closed
                adjacency[edge.a].append((edge.b, edge.capacity_kw))
                adjacency[edge.b].append((edge.a, edge.capacity_kw))

        supplied = np.zeros(len(self.graph.nodes), dtype=np.float32)
        for gen_idx in self.graph.generator_indices():
            capacity = self.graph.nodes[gen_idx].max_kw
            reachable = self._bfs_reachable(gen_idx, adjacency)
            loads = [i for i in reachable if self.graph.nodes[i].category == "load"]
            if not loads:
                continue
            per_load = capacity / len(loads)
            for load_idx in loads:
                node = self.graph.nodes[load_idx]
                supplied[load_idx] += min(per_load, node.demand_kw)
        return supplied

    def _bfs_reachable(self, start: int, adjacency: dict) -> List[int]:
        visited = set([start])
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor, _ in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return list(visited)

    def _violates_connectivity(self, switches: np.ndarray) -> bool:
        # Ensure every load is connected to at least one generator
        adjacency = {i: [] for i in range(len(self.graph.nodes))}
        for idx, edge in enumerate(self.graph.edges):
            if switches[idx] > 0.5:
                adjacency[edge.a].append(edge.b)
                adjacency[edge.b].append(edge.a)

        for load_idx in self.graph.load_indices():
            seen = set()
            queue = [load_idx]
            while queue:
                cur = queue.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if cur in self.graph.generator_indices():
                    break
                for nbr in adjacency[cur]:
                    queue.append(nbr)
            else:
                # BFS exhausted without finding generator
                return True
        return False

    def _overloaded(self, switches: np.ndarray, supplied: np.ndarray) -> bool:
        # Check line thermal capacity based on total power going through each edge
        for idx, edge in enumerate(self.graph.edges):
            if switches[idx] < 0.5:
                continue
            a, b = edge.a, edge.b
            flow = abs(supplied[a] - supplied[b])
            if flow - 1e-6 > edge.capacity_kw:
                return True
        return False

    def _reward(self) -> float:
        loads = self.graph.load_indices()
        load_supplied = self.state.supplied[loads]
        load_nodes = [self.graph.nodes[i] for i in loads]
        weighted_coverage = sum(
            node.priority * (supplied / (node.demand_kw + 1e-6)) for node, supplied in zip(load_nodes, load_supplied)
        )
        reward = self.cfg.base_reward_scale * weighted_coverage

        if self._violates_connectivity(self.state.switches):
            reward -= self.cfg.island_penalty
        if self._overloaded(self.state.switches, self.state.supplied):
            reward -= self.cfg.overload_penalty
        reward -= self.cfg.switch_cost * np.count_nonzero(self.state.switches != np.array([e.normally_closed for e in self.graph.edges]))
        return reward


# --- convenience factory -------------------------------------------------

def canonical_graph() -> ShipPowerGraph:
    """Returns a small MVDC-inspired topology with priorities from the paper."""
    nodes = [
        Node("G1", "generator", priority=0, demand_kw=0, max_kw=9.0),
        Node("G2", "generator", priority=0, demand_kw=0, max_kw=9.0),
        Node("L1", "load", priority=3, demand_kw=2.0),
        Node("L2", "load", priority=3, demand_kw=2.0),
        Node("L3", "load", priority=2, demand_kw=1.5),
        Node("L4", "load", priority=2, demand_kw=1.5),
        Node("L5", "load", priority=1, demand_kw=1.0),
        Node("L6", "load", priority=1, demand_kw=1.0),
    ]

    edges = [
        Edge(0, 2, capacity_kw=5.0, normally_closed=True, switchable=True),
        Edge(0, 3, capacity_kw=5.0, normally_closed=False, switchable=True),
        Edge(1, 4, capacity_kw=5.0, normally_closed=True, switchable=True),
        Edge(1, 5, capacity_kw=5.0, normally_closed=False, switchable=True),
        Edge(2, 4, capacity_kw=3.0, normally_closed=True, switchable=True),
        Edge(3, 5, capacity_kw=3.0, normally_closed=True, switchable=True),
        Edge(2, 6, capacity_kw=1.5, normally_closed=True, switchable=True),
        Edge(4, 7, capacity_kw=1.5, normally_closed=True, switchable=True),
        Edge(6, 7, capacity_kw=1.0, normally_closed=False, switchable=True),
    ]
    return ShipPowerGraph(nodes, edges)


def default_env(seed: int = 0) -> ShipPowerEnv:
    return ShipPowerEnv(canonical_graph(), EnvConfig(seed=seed))
