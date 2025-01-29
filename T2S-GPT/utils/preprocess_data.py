import os
import numpy as np

def normalized_data(input_skeleton, reference_skeleton):
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

    return scaled_skeleton

def standardized_data(input_skeleton, global_min, global_max, NUM_JOINT):
    scaled_skeleton = 2*((input_skeleton.reshape(input_skeleton.shape[0], -1) - global_min) / (global_max - global_min + 1e-8)).reshape(-1, NUM_JOINT, 3) - 1
    return scaled_skeleton

def compute_global_stats(all_scaled_skeletons):
    all_joint_values = []
    for input_skeleton in all_scaled_skeletons:
        if len(all_joint_values) == 0:
            all_joint_values = input_skeleton
        else:
            all_joint_values = np.concatenate((all_joint_values, input_skeleton), axis=0)

    all_joint_values = all_joint_values.reshape(all_joint_values.shape[0], -1) # (frame all vdo, joint*3)
    global_min = np.min(all_joint_values, axis=0) # Global min of each x,y,z joint (1 Joint == 3 min values)
    global_max = np.max(all_joint_values, axis=0)

    print("Global Stats",global_min.shape, global_max.shape)
    return global_min, global_max

def batch_process(input_dir, reference_dir, output_file):
    reference_skeleton = np.load(reference_dir)
    all_scaled_skeletons = []
    NUM_JOINT = reference_skeleton.shape[1]

    for skeleton_file in sorted(os.listdir(input_dir)): 
        input_skeleton = np.load(os.path.join(input_dir, skeleton_file))
        scaled_skeleton = normalized_data(input_skeleton, reference_skeleton)
        all_scaled_skeletons.append(scaled_skeleton)
    
    # Compute global statistics
    global_min, global_max = compute_global_stats(all_scaled_skeletons)

    with open(output_file, 'w') as f:
        for scaled_skeleton in all_scaled_skeletons:
            standardized_skeleton = standardized_data(scaled_skeleton, global_min, global_max, NUM_JOINT)
            if((standardized_skeleton.shape[1] != NUM_JOINT) |
                (standardized_skeleton.shape[2] != 3) |
                (standardized_skeleton.min() < -1) | 
                (standardized_skeleton.max() > 1)):
                print("Error: Value of standardized_skeleton is not correct!")
                exit(1)
            standardized_skeleton_str = ' '.join(map(str, standardized_skeleton.flatten()))
            f.write(standardized_skeleton_str + '\n')


batch_process(
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton',
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/raw_skeleton/01June_2010_Tuesday_tagesschau-5002-deblurred-with-BIN.npy',
    '/Users/iamkittitat/Desktop/2110488-Capstone-Text2Sign/T2S-GPT/data/scaled_skeleton/dev.skels'
)
