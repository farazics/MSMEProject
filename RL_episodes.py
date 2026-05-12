import time
import csv
import math
import random
import os

import numpy as np
import carla

from adaptive_controller import AdaptiveController


def get_vehicle_state(vehicle):
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return {
        'x': tf.location.x,
        'y': tf.location.y,
        'vx': vel.x,
        'vy': vel.y,
        'yaw': math.radians(tf.rotation.yaw)
    }


def get_next_run_filename(prefix="RL_run", folder="."):
    """Auto-increment run number: RL_run1.csv, RL_run2.csv, ..."""
    i = 1
    while os.path.exists(os.path.join(folder, f"{prefix}{i}.csv")):
        i += 1
    return os.path.join(folder, f"{prefix}{i}.csv")


# ===================== RL Agent =====================
class DiscreteRLAgent:
    def __init__(self,
                 v_bins=(0.0, 5.0, 10.0, 15.0, 25.0),
                 d_bins=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0),
                 dv_bins=(-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0),
                 actions=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
                 alpha=0.05,
                 gamma=0.99,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=1e-4,
                 q_init_path=None):
        self.v_bins = v_bins
        self.d_bins = d_bins
        self.dv_bins = dv_bins
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        shape = (len(v_bins) - 1,
                 len(d_bins) - 1,
                 len(dv_bins) - 1,
                 len(actions))

        if q_init_path is not None:
            try:
                q_loaded = np.load(q_init_path)
                if q_loaded.shape == shape:
                    self.q = q_loaded
                    print(f"Loaded Q-table from {q_init_path}")
                else:
                    print(f"Warning: Q-table shape mismatch {q_loaded.shape} != {shape}, using zeros.")
