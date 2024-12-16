import warnings
from typing import Sequence

import numpy as np

from skeleton_utils import compute_joint_tree


def position_to_absolute_angle(
    positional_joints: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to absolute angle format.
    PS. All angle is compared to the normal x, y, z axis.

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    absolute_angle_joints = np.zeros((positional_joints.shape[0], 3))

    if (positional_joints[root_joint] != 0).all():
        warnings.warn(f"The root joint {root_joint} is not at the origin (0, 0, 0)")

    # Compute the joint tree
    for bone_index in range(len(positional_joints)):
        if bone_index == root_joint:
            continue

        # Get the parent joint
        if bone_index not in joint_to_prev_joint_index:
            warnings.warn(f"Bone index {bone_index} not found in the skeleton model and will be skipped")
            continue
        parent_joint = joint_to_prev_joint_index[bone_index]

        # Get the position of the joint relative to the parent joint
        x, y, z = positional_joints[bone_index]
        prev_x, prev_y, prev_z = positional_joints[parent_joint]
        x, y, z = x - prev_x, y - prev_y, z - prev_z

        # Compute the spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        # Store the spherical coordinates
        absolute_angle_joints[bone_index] = np.array([r, theta, phi])

    return absolute_angle_joints


def absolute_angle_to_position(
    angle_joints: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts absolute angle format to positional format

    :param angle_joints: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance,
            theta = inclination angle (angle from the z-axis),
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model structure
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    positional_joints = np.zeros((angle_joints.shape[0], 3))

    if (angle_joints[root_joint] != 0).all():
        warnings.warn(f"The root joint {root_joint} is not at the origin (0, 0, 0)")

    # Compute the joint tree
    # Iterate using tree traversal to compute the position of the joints
    joint_tree = compute_joint_tree(skeleton_model)
    joint_index_queue = [root_joint]
    while joint_index_queue:
        parent_joint = joint_index_queue.pop(0)

        # Get the children joints
        children_joints = joint_tree.get(parent_joint, [])
        for child_joint in children_joints:
            r, theta, phi = angle_joints[child_joint]

            # Compute the position of the joint
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Get the parent joint
            if child_joint not in joint_to_prev_joint_index:
                warnings.warn(f"Bone index {child_joint} not found in the skeleton model")
                continue
            prev_joint = joint_to_prev_joint_index[child_joint]

            # Add the parent joint position
            x += positional_joints[prev_joint][0]
            y += positional_joints[prev_joint][1]
            z += positional_joints[prev_joint][2]

            if (positional_joints[child_joint] != 0).all():
                warnings.warn(
                    f"There are already values in the joint {child_joint} and will be overwritten to {x, y, z}"
                )
            positional_joints[child_joint] = np.array([x, y, z])

            # Add the child joint to the queue
            joint_index_queue.append(child_joint)

    return positional_joints


if __name__ == "__main__":
    from skeleton_utils.progressive_trans_model import (
        JOINT_TO_PREV_JOINT_INDEX,
        ROOT_JOINT,
        SKELETON_MODEL,
    )

    # original_joints = np.linspace(0, 50, 150).reshape(-1, 3)
    original_joints = np.random.rand(50, 3)
    original_joints = original_joints - original_joints[0]
    print(f"Original joints")
    print(original_joints)

    abs_angle = position_to_absolute_angle(original_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
    # print(f"Absolute angle joints")
    # print(abs_angle)

    pos_joints = absolute_angle_to_position(abs_angle, SKELETON_MODEL, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
    print("Positional joints")
    print(pos_joints)

    print("Is the original joints and positional joints the same? ->", np.allclose(original_joints, pos_joints))

    print("Checking for differences")
    for bone_index in range(50):
        if not np.allclose(original_joints[bone_index], pos_joints[bone_index]):
            print(f"Bone index {bone_index} is not the same")
            print("original\t:", original_joints[bone_index])
            print("positional\t:", pos_joints[bone_index])
            bone_path = [bone_index]
            current_bone = bone_index
            while current_bone != 0:
                if current_bone not in JOINT_TO_PREV_JOINT_INDEX:
                    break
                prev_bone = JOINT_TO_PREV_JOINT_INDEX[current_bone]
                bone_path.insert(0, prev_bone)
                current_bone = prev_bone
            print(" -> ".join(map(str, bone_path)))
            print()