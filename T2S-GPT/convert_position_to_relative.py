import torch

from utils.relative_angle_conversion import position_to_relative_angle
from utils.skeleton_utils.progressive_trans_model import JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT

pos_skeleton = "data/sampledata/test.skels"
output = "data/sampledata_relative/test.skels"
joint_sizes = 150

positional_skeleton = []
with open(pos_skeleton, "r") as f:
    data = f.readlines()
    data = [line.strip().split(" ") for line in data]
    data = [[float(val) for val in line] for line in data]
    data = [torch.tensor(line).reshape(-1, joint_sizes + 1)[:, :-1] for line in data]
    for line in data:
        positional_skeleton.append(line.numpy())

with open(output, "w") as f:
    for skeletons in positional_skeleton:
        to_save = []
        for frame_index, skeleton in enumerate(skeletons):
            skeleton = skeleton.reshape(-1, 3)
            relative_angle_skeleton = position_to_relative_angle(skeleton, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
            relative_angle_skeleton = relative_angle_skeleton.reshape(-1)
            to_save.extend(relative_angle_skeleton)
            to_save.append(frame_index)

        f.write(" ".join(map(str, to_save)) + "\n")