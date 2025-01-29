import os
import numpy as np

def scale_data(input_skeleton, reference_skeleton, global_mean, global_std):
    """
    Scale the input_skeleton using the reference_skeleton with normalization and global z-score standardization.
    Parameters:
        input_skeleton (np.ndarray): Shape (Frames, Joints, 3).
        reference_skeleton (np.ndarray): Same shape as input_skeleton to act as a reference (ONLY 1 ref in the entire dataset).
        global_mean (np.ndarray): Mean values for z-score standardization.
        global_std (np.ndarray): Standard deviation values for z-score standardization.
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

    # Global Standardization (Z-score)
    scaled_skeleton = (scaled_skeleton - global_mean) / global_std

    return scaled_skeleton

def compute_global_stats(input_dir):
    """
    Compute global mean and standard deviation for z-score standardization across the entire dataset.
    Parameters:
        input_dir (str): Directory containing the input skeleton files.
    Returns:
        tuple: Global mean and standard deviation.
    """
    all_joint_values = []

    for skeleton_file in os.listdir(input_dir):
        input_skeleton = np.load(os.path.join(input_dir, skeleton_file))
        joint_values = input_skeleton.reshape(-1, input_skeleton.shape[2])  # Flatten (Frames, Joints, 3) to (-1, 3)
        all_joint_values.append(joint_values)

    all_joint_values = np.vstack(all_joint_values)  # Combine all into a single array
    global_mean = np.mean(all_joint_values, axis=0)
    global_std = np.std(all_joint_values, axis=0)

    return global_mean, global_std

def batch_process(input_dir, reference_dir, output_file):
    reference_skeleton = np.load(reference_dir)

    # Compute global statistics
    global_mean, global_std = compute_global_stats(input_dir)

    with open(output_file, 'w') as f:
        for skeleton_file in os.listdir(input_dir):
            print(f"Processing {skeleton_file}")
            input_skeleton = np.load(os.path.join(input_dir, skeleton_file))
            scaled_skeleton = scale_data(input_skeleton, reference_skeleton, global_mean, global_std)
            
            scaled_skeleton_str = ' '.join(map(str, scaled_skeleton.flatten()))
            f.write(scaled_skeleton_str + '\n')


batch_process(
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton',
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton/01April_2010_Thursday_heute-6698-deblurred-with-BIN.npy',
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/scaled_skeleton/dev.skels'
)
