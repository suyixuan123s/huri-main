import numpy as np

# PARAMETERS
# ---------------------
# PURPLE RING CAO TYPE: 1
# BLUE CAP TYPE : 2
# WHITE CAP TYPE: 3


######## VISION SYSTEM ########
# Calibration Matrix
TCP2EYE_MAT = np.array([[0., 1., 0., -0.05074001],
                        [-1., 0., 0., 0.01460267],
                        [0., 0., 1., -0.08705009],
                        [0., 0., 0., 1.]])

# Test Tube Rack Detection
# Rack Height filter
HEIGHT_RANGE = (.030, 0.045)

######## Task PLANNING SYSTEM ########
GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

######## MOTION PLANNING SYSTEM ########
# Goal pose
GOAL_UP_DISTANCE = {1: .045,
                    2: .05,
                    3: .045, }

# Approach Distance
APPROACH_DISTANCE_LIST = {
    1: [.02, .07],
    2: [.06, .07],
    3: [.02, .07],
}

# Depart distance
DEPART_DISTANCE_LIST = [.15, .03]

######## Robot Motion ########
SPEED = 500
ACC = 4000

#### DEBUG ####
TOGGLE_RACK_LOCATOR = False
TOGGLE_YOLO = False
