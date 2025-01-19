import os
import numpy as np

def scale_data(input_skeleton, reference_skeleton):
    """
    Scale the input_skeleton using the reference_skeleton with normalization and z-score standardization.
    Parameters:
        input_skeleton (np.ndarray): Shape (Frames, Joints, 3).
        reference_skeleton (np.ndarray): Same shape as input_skeleton to act as a reference (ONLY 1 ref in the entire dataset).
    Returns:
        np.ndarray: Scaled skeleton with the same shape as input_skeleton.
    """

    # Normalization to reference_skeleton
    NOSE_JOINT = 520
    LEFT_SHOULDER_JOINT = 531
    RIGHT_SHOULDER_JOINT = 532

    scaled_skeleton = input_skeleton.copy()
    difference_distance = reference_skeleton[0, NOSE_JOINT, :] - input_skeleton[0, NOSE_JOINT, :]
    scaled_factor = (
        np.linalg.norm(reference_skeleton[0, LEFT_SHOULDER_JOINT, :] - reference_skeleton[0, RIGHT_SHOULDER_JOINT, :]) /
        np.linalg.norm(input_skeleton[0, LEFT_SHOULDER_JOINT, :] - input_skeleton[0, RIGHT_SHOULDER_JOINT, :]))

    for frame in range(input_skeleton.shape[0]):
        for joint in range(input_skeleton.shape[1]):
            scaled_skeleton[frame, joint, :] += difference_distance
            scaled_skeleton[frame, joint, :] = (
                reference_skeleton[0, NOSE_JOINT, :] +
                scaled_factor * (scaled_skeleton[frame, joint, :] - reference_skeleton[0, NOSE_JOINT, :])
            )

    # Standardization (Z-score)
    joint_values = scaled_skeleton.reshape(-1, scaled_skeleton.shape[2])
    mean = np.mean(joint_values, axis=0)
    std = np.std(joint_values, axis=0)
    scaled_skeleton = (scaled_skeleton - mean) / std

    return scaled_skeleton

def batch_process(input_dir, reference_dir, output_file):
    reference_skeleton = np.load(reference_dir)
    with open(output_file, 'w') as f:
        for skeleton_file in os.listdir(input_dir):
            input_skeleton = np.load(os.path.join(input_dir, skeleton_file))
            scaled_skeleton = scale_data(input_skeleton, reference_skeleton)
            
            scaled_skeleton_str = ' '.join(map(str, scaled_skeleton.flatten()))
            f.write(scaled_skeleton_str + '\n')


batch_process('/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton',
              '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton/01April_2010_Thursday_heute-6698.npy',
              '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/scaled_skeleton/dev.skels')