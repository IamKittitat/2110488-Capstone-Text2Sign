import warnings
from typing import Sequence

import numpy as np

from absolute_angle_conversion import absolute_angle_to_position, position_to_absolute_angle


def position_to_relative_angle(
    positional_joints: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to relative angle format.
    PS. All angle is the relative angle to the previous joint.

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which is the relative angle to the previous joint
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    # Convert the positional format to absolute angle
    absolute_angle_joints = position_to_absolute_angle(positional_joints, joint_to_prev_joint_index, root_joint)

    # Convert the absolute angle to relative angle
    relative_angle_joints = np.zeros((positional_joints.shape[0], 3))
    for bone_index in range(len(positional_joints)):
        if bone_index == root_joint:
            relative_angle_joints[bone_index] = absolute_angle_joints[bone_index]
            continue

        # Get the parent joint
        if bone_index not in joint_to_prev_joint_index:
            warnings.warn(f"Bone index {bone_index} does not have a parent joint")
            continue
        parent_joint = joint_to_prev_joint_index[bone_index]

        # Compute the relative angle
        r = absolute_angle_joints[bone_index][0]
        theta = absolute_angle_joints[bone_index][1] - absolute_angle_joints[parent_joint][1]
        phi = absolute_angle_joints[bone_index][2] - absolute_angle_joints[parent_joint][2]

        # Store the relative angle
        relative_angle_joints[bone_index] = np.array([r, theta, phi])

    return relative_angle_joints


def relative_angle_to_position(
    angle_joints: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    joint_to_child_joint_index: dict[int, Sequence[int]],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts relative angle format to positional format.
    PS. All angle is the relative angle to the previous joint.

    :param angle_joints: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which is the relative angle to the previous joint
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model which is a sequence of bone tuples
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param joint_to_child_joint_index: the mapping of joint index to child joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    """
    absolute_angle_joints = np.zeros((len(angle_joints), 3))

    # Convert the relative angle to absolute angle
    joint_queue = [root_joint]
    while joint_queue:
        current_joint = joint_queue.pop(0)
        if current_joint in joint_to_prev_joint_index:
            parent_joint = joint_to_prev_joint_index[current_joint]
            absolute_angle_joints[current_joint, 1:] = (
                angle_joints[current_joint, 1:] + absolute_angle_joints[parent_joint, 1:]
            )
            absolute_angle_joints[current_joint, 0] = angle_joints[current_joint, 0]
        else:
            absolute_angle_joints[current_joint] = angle_joints[current_joint]

        if current_joint in joint_to_child_joint_index:
            joint_queue.extend(joint_to_child_joint_index[current_joint])

    # Convert the absolute angle to positional format
    positional_joints = absolute_angle_to_position(
        absolute_angle_joints, skeleton_model, joint_to_prev_joint_index, root_joint
    )

    return positional_joints


if __name__ == "__main__":
    from skeleton_utils.progressive_trans_model import (
        JOINT_TO_CHILD_JOINTS_INDEX,
        JOINT_TO_PREV_JOINT_INDEX,
        ROOT_JOINT,
        SKELETON_MODEL,
    )

    # original_joints = np.linspace(0, 50, 150).reshape(-1, 3)
    original_joints = np.random.rand(50, 3)
    original_joints = original_joints - original_joints[0]
    print(f"Original joints")
    print(original_joints)

    rel_angle = position_to_relative_angle(original_joints, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
    print(f"Relative angle joints")
    print(rel_angle)

    pos_joints = relative_angle_to_position(
        rel_angle, SKELETON_MODEL, JOINT_TO_PREV_JOINT_INDEX, JOINT_TO_CHILD_JOINTS_INDEX, ROOT_JOINT
    )
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