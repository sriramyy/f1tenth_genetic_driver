import numpy as np

class GeneticParameters:
    def __init__(self):
# Follow the Gap Params
        self.BUBBLE_RADIUS = 130
        self.PREPROCESS_CONV_SIZE = 3
        self.BEST_POINT_CONV_SIZE = 120
        self.MAX_LIDAR_DIST = 7.0
        self.MAX_STEER_ABS = 0.698  # Approx 40 degrees

        # Speeds
        self.STRAIGHT_SPEED = 4.426
        self.CORNER_SPEED = 2.0
        self.SPEED_MAX = 5.5

        # Handling
        self.CENTER_BIAS_ALPHA = 0.35
        self.EDGE_GUARD_DEG = 12.0
        self.TTC_HARD_BRAKE = 0.55
        self.TTC_SOFT_BRAKE = 0.9
        self.FWD_WEDGE_DEG = 8.0
        self.STEER_SMOOTH_ALPHA = 0.449
        self.STEER_RATE_LIMIT = 0.14


    def update_from_dict(self, p_dict):
        """Used by the Genetic Algorithm to hot-swap values."""
        for key, value in p_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)