from __future__ import annotations

from gym.core import Env


class Node:
    def __init__(self, position, parent):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other: Node):
        return self.position == other.position

    def __lt__(self, other: Node):
        return self.f < other.f

    def __repr__(self):
        return f"({self.position[0]}, {self.position[1]}) -> {self.f}"


class AStarAgent:
    def __init__(self, env: Env, shape: tuple):
        positions = self.get_path(env.env.desc, (0, 0), (shape[0] - 1, shape[1] - 1))
        self.actions = self.generate_actions(positions, shape)

    @staticmethod
    def generate_actions(positions: list, shape: tuple) -> dict:
        actions = {}
        transitions = {
            (0, -1): 0,
            (1, 0): 1,
            (0, 1): 2,
            (-1, 0): 3,
        }

        for start, end in zip(positions[:-1], positions[1:]):
            diff = (end[0] - start[0], end[1] - start[1])
            actions[start[0] * shape[0] + start[1]] = transitions[diff]

        return actions

    def get_action(self, state):
        return self.actions[state]

    def get_path(self, board, start: tuple[int, int], end: tuple[int, int]) -> list:
        open_nodes = []
        closed_nodes = []

        start_node = Node(start, None)
        goal_node = Node(end, None)

        open_nodes.append(start_node)

        while open_nodes:
            open_nodes.sort()
            current_node = open_nodes.pop(0)
            closed_nodes.append(current_node)

            if current_node == goal_node:
                return reconstruct_path(current_node)

            x, y = current_node.position
            next_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for position in next_positions:
                x, y = position
                if not 0 <= x < board.shape[0] or not 0 <= y < board.shape[1]:
                    continue

                if board[x][y] == b'H':
                    continue

                neighbor_node = Node(position, current_node)
                neighbor_node.h = abs(position[0] - goal_node.position[0]) + abs(position[1] - goal_node.position[1])
                neighbor_node.g = current_node.g + 1
                neighbor_node.f = neighbor_node.h + neighbor_node.g

                if add_to_list(open_nodes, neighbor_node):
                    open_nodes.append(neighbor_node)

        return []


def add_to_list(nodes: list, node: Node) -> bool:
    for n in nodes:
        if n == node and node.f >= n.f:
            return False
    return True


def reconstruct_path(goal_node: Node):
    path = []

    current_node = goal_node
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent

    return path[::-1]
