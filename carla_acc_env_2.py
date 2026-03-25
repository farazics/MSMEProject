import os
import sys
import glob
import time
import numpy as np

# --- Force the exact CARLA client version to match your simulator (0.9.14) ---
# Adjust CARLA_ROOT if your path is different.
CARLA_ROOT = os.environ.get(
    "CARLA_ROOT",
    r"C:\Users\FarazShaikh(WovenbyT)\Downloads\CARLA_0.9.14\WindowsNoEditor"
)
egg_pattern = os.path.join(
    CARLA_ROOT, "PythonAPI", "carla", "dist",
    f"carla-0.9.14-py{sys.version_info.major}.{sys.version_info.minor}-win-amd64.egg"
)
candidates = glob.glob(egg_pattern)
if not candidates:  # wildcard fallback if minor version doesn't match filename
    candidates = glob.glob(os.path.join(
        CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-0.9.14-py3*.egg"
    ))
for egg in candidates:
    if egg not in sys.path:
        sys.path.append(egg)

import carla
import gymnasium as gym
from gymnasium import spaces

from adaptive_controller import AdaptiveController
from run_controller_5_EP import get_vehicle_state  # uses carla API correctly. :contentReference[oaicite:0]{index=0}


class CarlaACCEnv(gym.Env):
    """
    Gymnasium-style wrapper around CARLA + AdaptiveController.
    - Enforces CARLA 0.9.14 client (to match simulator 0.9.14)
    - Synchronous stepping (world.tick())
    - Same camera behavior as your original script
    - Minimal observation [gap, v_av, v_hdv] for plumbing
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 host: str = "localhost",
                 port: int = 2000,
                 dt: float = 0.1,
                 max_steps: int = 200,
                 town: str = None,     # optionally force a town name like "Town10HD_Opt"
                 sync_mode: bool = True):
        super().__init__()

        self.host = host
        self.port = port
        self.dt = dt
        self.max_steps = max_steps
        self.sync_mode = sync_mode
        self.town = town

        # --- Connect and (optionally) load town ---
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)

        if self.town:
            # Load the map explicitly if requested
            self.world = self.client.load_world(self.town)
        else:
            self.world = self.client.get_world()

        print(f"[CarlaACCEnv] Connected to world: {self.world.get_map().name}")

        # --- World settings: make stepping deterministic ---
        self._prev_settings = self.world.get_settings()
        if self.sync_mode:
            settings = carla.WorldSettings(
                synchronous_mode=True,
                fixed_delta_seconds=self.dt,
                no_rendering_mode=False
            )
            self.world.apply_settings(settings)

        # Controller (created in reset)
        self.controller = None

        # Actors
        self.av = None
        self.hdv = None

        # Spectator
        self.spectator = self.world.get_spectator()

        # Step counter
        self.step_idx = 0

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-3.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, 40.0, 40.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    # -------- Helpers --------

    def _cleanup_actors(self):
        for actor in [self.av, self.hdv]:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.av = None
        self.hdv = None

    def _compute_simple_obs(self, av_state, hdv_state):
        dx = hdv_state["x"] - av_state["x"]
        dy = hdv_state["y"] - av_state["y"]
        gap = float(np.hypot(dx, dy))
        v_av = float(np.hypot(av_state["vx"], av_state["vy"]))
        v_hdv = float(np.hypot(hdv_state["vx"], hdv_state["vy"]))
        return np.array([gap, v_av, v_hdv], dtype=np.float32)

    def _apply_hdv_behavior(self):
        # Same scripted leader behavior as your run_controller_5_EP.py. :contentReference[oaicite:1]{index=1}
        i = self.step_idx
        if i <= 50:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0))
        elif 51 <= i <= 100:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0))
        else:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))

    def _set_initial_spectator_view(self, av_tf: carla.Transform):
        if self.spectator is None:
            self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(
            av_tf.location + carla.Location(z=30.0),
            carla.Rotation(pitch=-90.0),
        ))

    def _update_chase_view(self):
        if self.spectator is None or self.av is None:
            return
        av_tf = self.av.get_transform()
        chase_loc = av_tf.location - av_tf.get_forward_vector() * 8.0 + carla.Location(z=5.0)
        chase_rot = carla.Rotation(pitch=-15.0, yaw=av_tf.rotation.yaw)
        self.spectator.set_transform(carla.Transform(chase_loc, chase_rot))

    # -------- Gymnasium API --------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clean previous
        self._cleanup_actors()
        self.step_idx = 0

        # Controller fresh state
        self.controller = AdaptiveController(
            time_headway=1.8,
            standstill_gap=5.0,
            alpha=1.0,
            dt=self.dt,
        )

        # Spawn points
        spawns = self.world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points found in CARLA map.")
        av_tf = spawns[0]

        # Leader 25 m ahead (same as your script). :contentReference[oaicite:2]{index=2}
        fwd = av_tf.get_forward_vector()
        hdv_location = av_tf.location + fwd * 25.0
        hdv_tf = carla.Transform(hdv_location, av_tf.rotation)

        # Blueprints
        bp_lib = self.world.get_blueprint_library()
        av_bp = bp_lib.find("vehicle.tesla.model3")
        hdv_bp = bp_lib.filter("vehicle.audi.tt")[0]

        # Spawn
        self.av = self.world.try_spawn_actor(av_bp, av_tf)
        self.hdv = self.world.try_spawn_actor(hdv_bp, hdv_tf)
        if self.av is None or self.hdv is None:
            self._cleanup_actors()
            raise RuntimeError("Could not spawn AV/HDV in reset().")

        # Ensure physics is on
        self.av.set_simulate_physics(True)
        self.hdv.set_simulate_physics(True)

        # Top-down view at reset
        self._set_initial_spectator_view(av_tf)

        # Tick once after spawn to ensure transforms/physics settle
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Initial obs (after one tick)
        av_state = get_vehicle_state(self.av)
        hdv_state = get_vehicle_state(self.hdv)
        obs = self._compute_simple_obs(av_state, hdv_state)

        info = {}
        return obs, info

    def step(self, action):
        self.step_idx += 1

        a_rl = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Current states
        av_state = get_vehicle_state(self.av)
        hdv_state = get_vehicle_state(self.hdv)

        # Leader scripted behavior
        self._apply_hdv_behavior()

        # Safe longitudinal command for AV via AdaptiveController (RL is nominal). :contentReference[oaicite:3]{index=3}
        a_cmd = self.controller.compute_control(av_state, hdv_state, a_rl)

        # Map accel -> throttle/brake (same mapping you use). :contentReference[oaicite:4]{index=4}
        throttle_cmd = max(a_cmd, 0.0)
        brake_cmd = max(-a_cmd, 0.0)
        self.av.apply_control(carla.VehicleControl(throttle=throttle_cmd, brake=brake_cmd, steer=0.0))

        # Update spectator chase view
        self._update_chase_view()

        # Advance world deterministically
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # New obs
        av_state_new = get_vehicle_state(self.av)
        hdv_state_new = get_vehicle_state(self.hdv)
        obs = self._compute_simple_obs(av_state_new, hdv_state_new)

        # Dummy reward + time limit (we replace next step)
        reward = 0.0
        terminated = False
        truncated = self.step_idx >= self.max_steps

        info = {"a_rl": a_rl, "a_cmd": a_cmd, "throttle": throttle_cmd, "brake": brake_cmd}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._cleanup_actors()
        # Restore previous world settings
        try:
            if self._prev_settings is not None:
                self.world.apply_settings(self._prev_settings)
        except Exception:
            pass
        time.sleep(0.5)