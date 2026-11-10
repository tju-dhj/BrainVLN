"""R2R-CE action space and the VLN-CE teacher-forcing hyperparameters."""

from __future__ import annotations

STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
ACTION_NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
SIM_ACTIONS = {
    MOVE_FORWARD: "move_forward",
    TURN_LEFT: "turn_left",
    TURN_RIGHT: "turn_right",
}

SUCCESS_DISTANCE = 3.0
FORWARD_STEP = 0.25
TURN_ANGLE = 15.0
MAX_EPISODE_STEPS = 500
RGB_SIZE = 224
HFOV = 90.0
DEPTH_SCALE = 1000.0
SENSOR_HEIGHT = 1.25
AGENT_HEIGHT = 1.5
AGENT_RADIUS = 0.1
PAD_INDEX = 0

# VLN-CE R2R teacher forcing: Adam 2.5e-4, inflection coefficient 3.2.
IL_LR = 2.5e-4
IL_EPOCHS = 30
IL_BATCH_SIZE = 2
INFLECTION_COEF = 3.2
GRAD_CLIP = 0.5
INSTRUCTION_HIDDEN = 128

DEFAULT_DATASET = "storage/datasets/R2R_VLNCE_v1-3_preprocessed"
DEFAULT_SCENES = "data/scene_datasets"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
