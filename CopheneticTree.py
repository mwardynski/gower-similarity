from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass()
class CopheneticTreeNode:
    right: Optional
    left: Optional
    parent: Optional
    label: Optional[int]
    height: Optional[np.float64]


class CopheneticTree:
    def __init__(self, dcoord: np.ndarray, labels: np.ndarray):
        self.dcoord = dcoord
        self.labels = labels
        self.head = CopheneticTreeNode(None, None, None, None, dcoord[-1][1])
        self.leaves = [None for _ in range(len(labels))]

    def build_tree(self, curr_node: CopheneticTreeNode, index):
        curr_index = index
        curr_node.left = CopheneticTreeNode(
            None, None, curr_node, None, self.dcoord[curr_index][0]
        )
        curr_node.right = CopheneticTreeNode(
            None, None, curr_node, None, self.dcoord[curr_index][3]
        )

        if not np.isclose(self.dcoord[curr_index][3], 0.0):
            index -= 1
            index = self.build_tree(curr_node.right, index)

        if not np.isclose(self.dcoord[curr_index][0], 0.0):
            index -= 1
            index = self.build_tree(curr_node.left, index)

        return index

    def get_left_node(self):
        curr_node = self.head

        while curr_node.left is not None:
            curr_node = curr_node.left

        return curr_node

    def get_next_right_node(self, curr_node: CopheneticTreeNode):
        last_node = None

        while (
            curr_node.right is None
            or last_node is curr_node.right
            or last_node is None
        ):
            if curr_node.parent is None:
                return None

            last_node = curr_node
            curr_node = curr_node.parent

        curr_node = curr_node.right

        while curr_node.left is not None:
            curr_node = curr_node.left

        return curr_node

    def label_tree(self, left_node: CopheneticTreeNode, index: int):
        curr_node = left_node
        while curr_node is not None:
            curr_node.label = self.labels[index]
            self.leaves[index] = curr_node
            index += 1
            curr_node = self.get_next_right_node(curr_node)

    def print_tree(self, curr_node: CopheneticTreeNode, indent: int):
        print(" " * indent, f"{curr_node.height}, {curr_node.label}")

        if curr_node.left is not None:
            self.print_tree(curr_node.left, indent + 1)

        if curr_node.right is not None:
            self.print_tree(curr_node.right, indent + 1)

    def fill_subtree(
        self,
        dist_matrix: np.ndarray,
        instance_id: int,
        curr_node: CopheneticTreeNode,
        height: np.float64,
    ):
        if curr_node.label is not None:
            dist_matrix[instance_id][curr_node.label] = height
            dist_matrix[curr_node.label][instance_id] = height
            return

        self.fill_subtree(dist_matrix, instance_id, curr_node.left, height)
        self.fill_subtree(dist_matrix, instance_id, curr_node.right, height)

    def get_distance_for_instance(
        self, dist_matrix: np.ndarray, instance_index: int
    ):
        instance_id = self.labels[instance_index]

        last_node = None
        current_node = self.leaves[instance_index]

        while current_node is not None:
            last_node = current_node
            current_node = current_node.parent
            if current_node is None:
                return

            while current_node.right is last_node:
                last_node = current_node
                current_node = current_node.parent
                if current_node is None:
                    return

            self.fill_subtree(
                dist_matrix,
                instance_id,
                current_node.right,
                current_node.height,
            )


def get_cophenetic_distance_matrix(dcoord: np.ndarray, labels: np.ndarray):

    coph = CopheneticTree(dcoord, labels)

    coph.build_tree(coph.head, len(dcoord) - 1)

    left_node = coph.get_left_node()

    coph.label_tree(left_node, 0)

    # coph.print_tree(coph.head, 0)

    dist_matrix = np.zeros((len(labels), len(labels)))

    for i in range(len(labels)):
        coph.get_distance_for_instance(dist_matrix, i)

    # print(dist_matrix)
    return dist_matrix
