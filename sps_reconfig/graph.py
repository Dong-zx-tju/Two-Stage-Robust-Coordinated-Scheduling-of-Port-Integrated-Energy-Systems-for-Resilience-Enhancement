"""
Graph definitions for the simplified ship power system used in the
DQN-MASK reconfiguration demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class Node:
    """Represents a bus in the shipboard power system graph."""

    name: str
    category: str  # "load" or "generator"
    priority: int = 1
    demand_kw: float = 0.0
    max_kw: float = 0.0


@dataclass
class Edge:
    """Connection between buses.

    Attributes:
        a: index of the first node.
        b: index of the second node.
        capacity_kw: thermal capacity of the line.
        normally_closed: whether the line is normally closed.
        switchable: whether the reconfiguration agent can toggle this line.
    """

    a: int
    b: int
    capacity_kw: float
    normally_closed: bool
    switchable: bool


class ShipPowerGraph:
    """A simple graph container with helper lookup functions."""

    def __init__(self, nodes: Sequence[Node], edges: Sequence[Edge]):
        self.nodes: List[Node] = list(nodes)
        self.edges: List[Edge] = list(edges)
        self.adjacency: Dict[int, List[int]] = {i: [] for i in range(len(self.nodes))}
        for idx, edge in enumerate(self.edges):
            self.adjacency[edge.a].append(idx)
            self.adjacency[edge.b].append(idx)

    def generator_indices(self) -> List[int]:
        return [i for i, node in enumerate(self.nodes) if node.category == "generator"]

    def load_indices(self) -> List[int]:
        return [i for i, node in enumerate(self.nodes) if node.category == "load"]

    def edge_endpoints(self, edge_idx: int) -> tuple[int, int]:
        edge = self.edges[edge_idx]
        return edge.a, edge.b
