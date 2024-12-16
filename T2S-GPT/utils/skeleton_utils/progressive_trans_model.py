ROOT_JOINT = 0
# skeleton model
# must in format of (start, end, bone)
# bone is the index of the bone
# start and end are the index of the joints
# and the structure must be a tree
# this is the structure of the skeleton model
SKELETON_MODEL = (
    # head
    (0, 1, 0),
    # left shoulder
    (1, 2, 1),
    # left arm
    (2, 3, 2),
    # (3, 4, 3), #
    # Changed to avoid wrist, go straight to hands
    (3, 29, 3),
    # right shoulder
    (1, 5, 1),
    # right arm
    (5, 6, 2),
    # (6, 7, 3), #
    # Changed to avoid wrist, go straight to hands
    (6, 8, 3),
    # left hand - wrist
    # (7, 8, 4), #
    # left hand - palm
    (8, 9, 5),
    (8, 13, 9),
    (8, 17, 13),
    (8, 21, 17),
    (8, 25, 21),
    # left hand - 1st finger
    (9, 10, 6),
    (10, 11, 7),
    (11, 12, 8),
    # left hand - 2nd finger
    (13, 14, 10),
    (14, 15, 11),
    (15, 16, 12),
    # left hand - 3rd finger
    (17, 18, 14),
    (18, 19, 15),
    (19, 20, 16),
    # left hand - 4th finger
    (21, 22, 18),
    (22, 23, 19),
    (23, 24, 20),
    # left hand - 5th finger
    (25, 26, 22),
    (26, 27, 23),
    (27, 28, 24),
    # right hand - wrist
    # (4, 29, 4), #
    # right hand - palm
    (29, 30, 5),
    (29, 34, 9),
    (29, 38, 13),
    (29, 42, 17),
    (29, 46, 21),
    # right hand - 1st finger
    (30, 31, 6),
    (31, 32, 7),
    (32, 33, 8),
    # right hand - 2nd finger
    (34, 35, 10),
    (35, 36, 11),
    (36, 37, 12),
    # right hand - 3rd finger
    (38, 39, 14),
    (39, 40, 15),
    (40, 41, 16),
    # right hand - 4th finger
    (42, 43, 18),
    (43, 44, 19),
    (44, 45, 20),
    # right hand - 5th finger
    (46, 47, 22),
    (47, 48, 23),
    (48, 49, 24),
)

JOINT_TO_PREV_JOINT_INDEX = dict((bone[1], bone[0]) for bone in SKELETON_MODEL)

JOINT_TO_CHILD_JOINTS_INDEX = {}
for bone in SKELETON_MODEL:
    if bone[0] not in JOINT_TO_CHILD_JOINTS_INDEX:
        JOINT_TO_CHILD_JOINTS_INDEX[bone[0]] = []
    JOINT_TO_CHILD_JOINTS_INDEX[bone[0]].append(bone[1])