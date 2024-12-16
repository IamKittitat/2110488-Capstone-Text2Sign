from typing import Sequence


# get bone colour given index
def get_bone_colour(skeleton: int, j: int) -> tuple[int, int, int]:
    """
    This function returns the colour of the bone given the index

    :param skeleton: The skeleton model structure
    :param j: The index of the bone
    :return: The colour of the bone in the format `(B, G, R)`
    """
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)

    return c


def compute_joint_tree(skeleton: Sequence[tuple[int, int, int]]) -> dict[int, list[int]]:
    """
    This function computes the joint tree structure from the skeleton model

    :param skeleton: The skeleton model structure
    :return: The joint tree structure in the format `{parent: [child1, child2, ...]}`
    """
    joint_tree = {}
    for parent, child, _ in skeleton:
        if parent not in joint_tree:
            joint_tree[parent] = []
        joint_tree[parent].append(child)

    return joint_tree