import os
import sys
import time
import numpy as np

# =========================
# Load CARLA 0.9.14 Python egg (hard-coded path)
# =========================

CARLA_EGG = r"C:\Users\FarazShaikh(WovenbyT\Downloads\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.14-py3.7-win-amd64.egg"

if not os.path.isfile(CARLA_EGG):
    raise RuntimeError(
        f"CARLA egg not found at:\n  {CARLA_EGG}\n"
        f"Please check that this path is correct."
    )

# If carla was imported earlier in this process (by something else), remove it
if "carla" in sys.modules:
    del sys.modules["carla"]

# Put the egg at the FRONT of sys.path so Python loads CARLA from it
if CARLA_EGG not in sys.path:
    sys.path.insert(0, CARLA_EGG)

print("[carla_acc_env] Using CARLA Python egg:", CARLA_EGG)

import carla
import gymnasium as gym
from gymnasium import spaces

from adaptive_controller import AdaptiveController
from run_controller_5_EP import get_vehicle_state


class CarlaACCEnv(gym.Env):
    """
    Gymnasium-style wrapper around CARLA + AdaptiveController.

    - Uses CARLA 0.9.14 client (matching your 0.9.14 simulator)
    - Synchronous stepping (world.tick()) for deterministic behavior
    - Same camera behavior as your original script:
        * top-down at reset
        * chase-view behind AV each step
    - For now: simple observation [gap, v_av, v_hdv] and dummy reward
      (we'll upgrade later to the full RL state and reward).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 host: str = "localhost",
                 port: int = 2000,
                 dt: float = 0.1,
                 max_steps: int = 200,
                 town: str = None,   # optionally force a specific map
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
        print(f"[CarlaACCEnv] Client API version: {self.client.get_client_version()}")
        print(f"[CarlaACCEnv] Server API version: {self.client.get_server_version()}")

        # --- World settings: make stepping deterministic (synchronous mode) ---
        self._prev_settings = self.world.get_settings()
        if self.sync_mode:
            settings = carla.WorldSettings(
                synchronous_mode=True,       # we control stepping explicitly
                fixed_delta_seconds=self.dt,
                no_rendering_mode=False
            )
            self.world.apply_settings(settings)

        # Controller (created in reset)
        self.controller: AdaptiveController | None = None

        # Actors
        self.av = None
        self.hdv = None

        # Spectator for camera
        self.spectator = self.world.get_spectator()

        # Step counter
        self.step_idx = 0

        # --- Gym spaces ---

        # Action: RL nominal acceleration in [-3, 3] m/s^2
        # self.action_space = spaces.Box(
        #     low=np.array([-3.0], dtype=np.float32),
        #     high=np.array([3.0], dtype=np.float32),
        #     dtype=np.float32,
        # )

        self.action_space = spaces.Box(
        low=np.array([-1.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        dtype=np.float32,
        )

        # Observation (for now): [gap, v_av, v_hdv]
        # obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # obs_high = np.array([100.0, 40.0, 40.0], dtype=np.float32)
        # self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        obs_low = np.array([0.0, 0.0, 0.0, -40.0, 0.0], dtype=np.float32)
        obs_high = np.array([100.0, 40.0, 40.0, 40.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
                # --- Reward weights (tunable) ---
        self.target_headway = 2.0      # s
        self.headway_tol = 1.0         # soft tolerance around target

        self.w_headway = 0.5           # headway tracking weight
        self.w_safety = 10.0           # safety penalty weight
        self.w_accel = 0.1             # acceleration smoothness weight
        self.w_progress = 0.05         # encourage AV speed
        self.w_lazy = 0.2              # penalize standing still while HDV moves

        self.min_gap = 3.0             # [m] "too close" threshold for reward


    # -------- Helper methods --------

    def _cleanup_actors(self):
        """Destroy AV and HDV if they exist."""
        for actor in [self.av, self.hdv]:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.av = None
        self.hdv = None

    # def _compute_simple_obs(self, av_state, hdv_state):
    #     """Current minimal observation: [gap, v_av, v_hdv]."""
    #     dx = hdv_state["x"] - av_state["x"]
    #     dy = hdv_state["y"] - av_state["y"]
    #     gap = float(np.hypot(dx, dy))

    #     v_av = float(np.hypot(av_state["vx"], av_state["vy"]))
    #     v_hdv = float(np.hypot(hdv_state["vx"], hdv_state["vy"]))

    #     obs = np.array([gap, v_av, v_hdv], dtype=np.float32)
    #     return obs


    def _compute_obs(self, av_state, hdv_state):
        """
        Full observation:
        gap       : Euclidean distance between AV and HDV [m]
        v_av      : AV speed [m/s]
        v_hdv     : HDV speed [m/s]
        rel_speed : v_hdv - v_av [m/s]
        headway   : gap / max(v_av, eps) [s], clipped to [0, 10]
        """
        dx = hdv_state["x"] - av_state["x"]
        dy = hdv_state["y"] - av_state["y"]
        gap = float(np.hypot(dx, dy))

        v_av = float(np.hypot(av_state["vx"], av_state["vy"]))
        v_hdv = float(np.hypot(hdv_state["vx"], hdv_state["vy"]))

        rel_speed = v_hdv - v_av

        eps = 0.1  # avoid division by zero at standstill
        if v_av < eps:
            headway = 10.0  # treat as "very large" headway when AV is basically stopped
        else:
            headway = gap / v_av
            headway = max(0.0, min(headway, 10.0))

        obs = np.array([gap, v_av, v_hdv, rel_speed, headway], dtype=np.float32)
        return obs
    
    def _compute_reward(self, obs, a_cmd, nominal_acc):
        """
        Reward components:

        obs = [gap, v_av, v_hdv, rel_speed, headway]

        - Headway tracking:
            penalize deviation from target_headway
        - Safety:
            big penalty if gap < min_gap
        - Smoothness:
            penalize large |a_cmd|
        - Progress:
            reward AV speed, and penalize "lazy" behavior when HDV moves but AV doesn't
        """
        gap, v_av, v_hdv, rel_speed, headway = obs

        # --- Headway tracking ---
        # Squared error around target_headway
        headway_error = headway - self.target_headway
        r_headway = - self.w_headway * (headway_error ** 2)

        # --- Safety term ---
        if gap < self.min_gap:
            # Quadratic penalty when closer than min_gap
            r_safety = - self.w_safety * ((self.min_gap - gap) ** 2)
        else:
            r_safety = 0.0

        # --- Smoothness / comfort ---
        r_accel = - self.w_accel * (a_cmd ** 2)

        # --- Progress / avoid laziness ---
        # Reward moving forward
        r_progress = self.w_progress * v_av

        # Extra penalty if HDV is moving but AV is basically stopped
        if v_hdv > 0.5 and v_av < 0.1:
            r_progress -= self.w_lazy * gap

        reward = r_headway + r_safety + r_accel + r_progress
        return float(reward)



    def _apply_hdv_behavior(self):
        """
        Scripted HDV (leader) behavior, same as your run_controller_5_EP.py:
        - accelerate (0-50 steps)
        - reduce throttle (51-100)
        - brake to stop (after 100)
        """
        i = self.step_idx

        if i <= 50:
            self.hdv.apply_control(
                carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0)
            )
        elif 51 <= i <= 100:
            self.hdv.apply_control(
                carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0)
            )
        else:
            self.hdv.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0)
            )

    def _set_initial_spectator_view(self, av_tf: carla.Transform):
        """Top-down camera over the AV at reset."""
        if self.spectator is None:
            self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(
            av_tf.location + carla.Location(z=30.0),
            carla.Rotation(pitch=-90.0),
        ))

    def _update_chase_view(self):
        """Chase-view camera behind the AV each step."""
        if self.spectator is None or self.av is None:
            return
        av_tf = self.av.get_transform()
        chase_loc = av_tf.location - av_tf.get_forward_vector() * 8.0 + carla.Location(z=5.0)
        chase_rot = carla.Rotation(pitch=-15.0, yaw=av_tf.rotation.yaw)
        self.spectator.set_transform(carla.Transform(chase_loc, chase_rot))

    # -------- Gymnasium API --------

    def reset(self, *, seed=None, options=None):
        """
        Spawn AV and HDV, initialize controller, set camera, and return initial (obs, info).
        """
        super().reset(seed=seed)

        # Clean previous actors
        self._cleanup_actors()
        self.step_idx = 0

        # New controller instance for fresh internal state
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

        # Leader (HDV) 25 m ahead along AV heading
        fwd = av_tf.get_forward_vector()
        hdv_location = av_tf.location + fwd * 25.0
        hdv_tf = carla.Transform(hdv_location, av_tf.rotation)

        # Blueprints
        bp_lib = self.world.get_blueprint_library()
        av_bp = bp_lib.find("vehicle.tesla.model3")
        hdv_bp = bp_lib.filter("vehicle.audi.tt")[0]

        # Spawn vehicles
        self.av = self.world.try_spawn_actor(av_bp, av_tf)
        self.hdv = self.world.try_spawn_actor(hdv_bp, hdv_tf)
        if self.av is None or self.hdv is None:
            self._cleanup_actors()
            raise RuntimeError("Could not spawn AV/HDV in reset().")

        # Ensure physics is on
        self.av.set_simulate_physics(True)
        self.hdv.set_simulate_physics(True)

        # Top-down camera at reset
        self._set_initial_spectator_view(av_tf)

        # Tick once after spawn so transforms/physics are valid
        self.world.tick()  # sync_mode is True by default

        # Initial observation
        av_state = get_vehicle_state(self.av)
        hdv_state = get_vehicle_state(self.hdv)
        obs = self._compute_obs(av_state, hdv_state)

        info: dict = {}
        return obs, info

    # def step(self, action):
    #     """
    #     One RL step:
    #       - interpret action as nominal acceleration a_rl
    #       - apply HDV scripted behavior
    #       - compute safe a_cmd via AdaptiveController
    #       - apply throttle/brake to AV
    #       - advance CARLA one tick
    #       - return (obs, reward, terminated, truncated, info)
    #     """
    #     self.step_idx += 1

    #     # Clip action to valid range
    #     # a_rl = float(np.clip(
    #     #     action[0],
    #     #     self.action_space.low[0],
    #     #     self.action_space.high[0],
    #     # ))

    #     # # Current states
    #     # av_state = get_vehicle_state(self.av)
    #     # hdv_state = get_vehicle_state(self.hdv)

    #     # # Scripted leader behavior
    #     # self._apply_hdv_behavior()

    #     # # Safe longitudinal command via AdaptiveController (RL provides nominal input)
    #     # a_cmd = self.controller.compute_control(av_state, hdv_state, a_rl)



    #     a_rl = float(np.clip(action[0], -1.0, 1.0))

    #     # Map [-1, 1] -> [0.0, 2.0] for nominal acceleration
    #     nominal_acc = 1.0 + a_rl  # center around 1.0 as before

    #     av_state = get_vehicle_state(self.av)
    #     hdv_state = get_vehicle_state(self.hdv)

    #     self._apply_hdv_behavior()

    #     a_cmd = self.controller.compute_control(av_state, hdv_state, nominal_acc)


    def step(self, action):
        """
        One RL step:

          - (For now) ignore action and use nominal_acc = 1.0
            to reproduce your original ACC behavior.
          - Apply HDV scripted behavior.
          - Compute safe a_cmd via AdaptiveController.
          - Apply throttle/brake to AV.
          - Advance CARLA one tick.
          - Compute reward and termination.
        """
        self.step_idx += 1

        # We still parse the action so shape errors are obvious,
        # but we ignore it for now. RL integration will change this.
        a_raw = float(np.clip(
            action[0],
            self.action_space.low[0],
            self.action_space.high[0],
        ))

        # Nominal acceleration (same as original script)
        nominal_acc = 1.0

        # Current states
        av_state = get_vehicle_state(self.av)
        hdv_state = get_vehicle_state(self.hdv)

        # Scripted leader behavior
        self._apply_hdv_behavior()

        # Safe longitudinal command via AdaptiveController
        a_cmd = self.controller.compute_control(av_state, hdv_state, nominal_acc)








        # Map accel -> throttle / brake
        throttle_cmd = max(a_cmd, 0.0)
        brake_cmd = max(-a_cmd, 0.0)

        # Apply control to AV
        self.av.apply_control(carla.VehicleControl(
            throttle=throttle_cmd,
            brake=brake_cmd,
            steer=0.0,
        ))

        # Update chase camera
        self._update_chase_view()

        # Advance CARLA one step
        self.world.tick()

        # New obs
        # av_state_new = get_vehicle_state(self.av)
        # hdv_state_new = get_vehicle_state(self.hdv)
        # obs = self._compute_obs(av_state_new, hdv_state_new)

        # # For now: dummy reward and time-limit truncation only
        # reward = 0.0
        # terminated = False
        # truncated = self.step_idx >= self.max_steps

        # info = {
        #     "a_rl": a_rl,
        #     "a_cmd": a_cmd,
        #     "throttle": throttle_cmd,
        #     "brake": brake_cmd,
        # }

        # return obs, reward, terminated, truncated, info

                # New obs (full state)
        av_state_new = get_vehicle_state(self.av)
        hdv_state_new = get_vehicle_state(self.hdv)
        obs = self._compute_obs(av_state_new, hdv_state_new)

        # Reward
        reward = self._compute_reward(obs, a_cmd, nominal_acc)

        # Termination: "near collision" if gap < 1.0 m
        gap = float(obs[0])
        terminated = gap < 1.0

        # Time-limit truncation
        truncated = self.step_idx >= self.max_steps

        info = {
            "a_raw": a_raw,
            "nominal_acc": nominal_acc,
            "a_cmd": a_cmd,
            "throttle": throttle_cmd,
            "brake": brake_cmd,
            "obs": obs,
        }

        return obs, reward, terminated, truncated, info




    def close(self):
        """Destroy actors and restore previous world settings."""
        self._cleanup_actors()
        try:
            if hasattr(self, "_prev_settings") and self._prev_settings is not None:
                self.world.apply_settings(self._prev_settings)
        except Exception:
            pass
        time.sleep(0.5)

if __name__ == "__main__":
    # Simple sanity check: create env, reset once, then close
    env = CarlaACCEnv(dt=0.1, max_steps=50)
    obs, info = env.reset()
    print("Initial obs from __main__ test:", obs)
    env.close()
