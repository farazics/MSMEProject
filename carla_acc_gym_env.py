# carla_acc_gym_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import time

from adaptive_controller import AdaptiveController


def get_vehicle_state(vehicle):
    """Extract position and velocity from CARLA vehicle."""
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return {
        "x": tf.location.x,
        "y": tf.location.y,
        "vx": vel.x,
        "vy": vel.y,
        "yaw": np.deg2rad(tf.rotation.yaw)
    }


class CarlaACCEnv(gym.Env):
    """
    Gym environment for adaptive cruise control using CARLA simulator.
    
    RL agent chooses nominal_acceleration each step.
    AdaptiveController enforces safety and smoothness.
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 client_host="localhost",
                 client_port=2000,
                 dt=0.1,
                 max_steps=200):
        """
        Initialize the environment.
        
        Args:
            client_host: CARLA server host
            client_port: CARLA server port
            dt: simulation timestep (seconds)
            max_steps: maximum steps per episode
        """
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0

        # Connect to CARLA once (reuse connection across episodes)
        self.client = carla.Client(client_host, client_port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        # Set fixed timestep (asynchronous mode)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = dt
        self.world.apply_settings(settings)

        self.ego: carla.Actor | None = None
        self.hdv: carla.Actor | None = None

        # Create controller (same as your run_controller script)
        self.controller = AdaptiveController(
            time_headway=1.8,
            standstill_gap=5.0,
            alpha=1.0,
            dt=dt
        )

        # Define observation space: [gap, v_av, v_hdv, rel_speed, headway]
        obs_low  = np.array([0.0,   0.0,   0.0,   -np.inf, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Define action space: a_rl in [-3.0, 3.0] m/s^2
        self.action_space = spaces.Box(
            low=np.array([-3.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32
        )

        # Reward function parameters
        self.h_des = 1.8      # desired headway (seconds)
        self.h_band = 0.7     # tolerance band for headway error
        self.w_h = 1.0        # weight on headway tracking
        self.a_ref = 1.5      # reference acceleration magnitude
        self.w_a = 0.05       # weight on comfort (smooth acceleration)
        self.d_safe_min = 5.0 # minimum safe gap (meters)
        self.w_safe = 15.0    # weight on safety penalty
        self.d_far = 40.0     # distance threshold for "too far" penalty
        self.w_far = 0.5      # weight on progress penalty

    # ============ CORE GYM METHODS ============

    def reset(self, *, seed=None, options=None):
        """Reset environment: spawn new AV and HDV, return initial observation."""
        super().reset(seed=seed)
        self.step_count = 0

        # Clean up from previous episode
        self._destroy_actors()
        
        # Spawn new AV and HDV
        self._spawn_actors()

        # Get initial observation
        obs = self._get_obs()
        info: dict = {}
        
        return obs, info

    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: numpy array [a_rl] where a_rl is the RL-chosen acceleration
        
        Returns:
            obs: new observation
            reward: scalar reward
            done: whether episode is done
            info: dictionary with debug info
        """
        self.step_count += 1

        # Ensure action is within bounds
        a_rl = float(np.clip(action[0],
                             self.action_space.low[0],
                             self.action_space.high[0]))

        # === SCRIPTED HDV BEHAVIOR (copied from your run_controller script) ===
        # Steps 1-50: accelerate (throttle=0.5)
        # Steps 51-100: cruise (throttle=0.3)
        # Steps 101+: brake (brake=0.6)
        if self.step_count <= 50:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0))
        elif 51 <= self.step_count <= 100:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0))
        else:
            self.hdv.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))

        # Get current state (before AV control)
        gap, v_av, v_hdv, rel_speed, headway = self._measure_state()

        # === RL + ADAPTIVE CONTROLLER ===
        # Get vehicle states
        av_state = get_vehicle_state(self.ego)
        hdv_state = get_vehicle_state(self.hdv)

        # RL chooses nominal acceleration, controller enforces safety
        a_cmd = self.controller.compute_control(
            av_state=av_state,
            hdv_state=hdv_state,
            nominal_acceleration=a_rl  # <-- RL decides this each step
        )

        # Convert acceleration to throttle/brake
        throttle_cmd, brake_cmd = self._accel_to_throttle_brake(a_cmd)

        # Apply control to AV
        self.ego.apply_control(carla.VehicleControl(
            throttle=throttle_cmd,
            brake=brake_cmd,
            steer=0.0
        ))

        # === STEP WORLD ===
        # In asynchronous mode, sleep to let physics process
        time.sleep(self.dt * 0.5)

        # === MEASURE NEW STATE ===
        gap, v_av, v_hdv, rel_speed, headway = self._measure_state()
        obs = np.array([gap, v_av, v_hdv, rel_speed, headway], dtype=np.float32)

        # === COMPUTE REWARD ===
        reward = self._compute_reward(gap, v_av, v_hdv, rel_speed, headway, a_rl)

        # === DETERMINE IF EPISODE ENDS ===
        done = False
        collision = self._check_collision(gap)
        
        if collision:
            reward += -50.0  # large collision penalty
            done = True

        if self.step_count >= self.max_steps:
            done = True

        if gap < 0.0:  # something very wrong
            reward += -20.0
            done = True

        # Build info dict for debugging
        info: dict = {
            "gap": float(gap),
            "v_av": float(v_av),
            "v_hdv": float(v_hdv),
            "rel_speed": float(rel_speed),
            "headway": float(headway),
            "a_rl": float(a_rl),
            "a_cmd": float(a_cmd),
            "collision": collision,
            "throttle": float(throttle_cmd),
            "brake": float(brake_cmd),
        }

        return obs, reward, done, info

    # ============ INTERNAL HELPER METHODS ============

    def _spawn_actors(self) -> None:
        """Spawn AV and HDV at the start of an episode."""
        spawns = self.world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points found in map.")

        # AV spawns at first spawn point
        av_tf = spawns[0]
        
        # HDV spawns 25m ahead of AV along road direction
        fwd = av_tf.get_forward_vector()
        hdv_location = av_tf.location + fwd * 25.0
        hdv_tf = carla.Transform(hdv_location, av_tf.rotation)

        # Get vehicle blueprints
        bp_lib = self.world.get_blueprint_library()
        av_bp = bp_lib.find('vehicle.tesla.model3')
        hdv_bp = bp_lib.filter('vehicle.audi.tt')[0]

        # Try to spawn
        self.ego = self.world.try_spawn_actor(av_bp, av_tf)
        self.hdv = self.world.try_spawn_actor(hdv_bp, hdv_tf)

        if self.ego is None or self.hdv is None:
            if self.ego is not None:
                self.ego.destroy()
            if self.hdv is not None:
                self.hdv.destroy()
            raise RuntimeError("Could not spawn AV and/or HDV.")

    def _destroy_actors(self) -> None:
        """Clean up vehicles at end of episode."""
        if self.ego is not None:
            self.ego.destroy()
            self.ego = None
        if self.hdv is not None:
            self.hdv.destroy()
            self.hdv = None

    def _measure_state(self) -> tuple:
        """Measure current gap, velocities, and compute derived quantities."""
        av_state = get_vehicle_state(self.ego)
        hdv_state = get_vehicle_state(self.hdv)

        # Compute gap (Euclidean distance)
        dx = hdv_state["x"] - av_state["x"]
        dy = hdv_state["y"] - av_state["y"]
        gap = float(np.hypot(dx, dy))

        # Compute velocities
        v_av = float(np.hypot(av_state["vx"], av_state["vy"]))
        v_hdv = float(np.hypot(hdv_state["vx"], hdv_state["vy"]))
        rel_speed = v_hdv - v_av

        # Compute time headway (gap / speed)
        v_for_headway = max(v_av, 0.1)  # avoid division by zero
        headway = gap / v_for_headway if gap >= 0.0 else 0.0

        return gap, v_av, v_hdv, rel_speed, headway

    def _get_obs(self) -> np.ndarray:
        """Return current observation as numpy array."""
        gap, v_av, v_hdv, rel_speed, headway = self._measure_state()
        return np.array([gap, v_av, v_hdv, rel_speed, headway], dtype=np.float32)

    def _accel_to_throttle_brake(self, a_cmd: float) -> tuple:
        """
        Convert acceleration command to throttle/brake.
        More aggressive: scale smaller to get actual motion.
        """
        if a_cmd >= 0.0:
            # Positive acceleration -> throttle
            # Scale: a_cmd of 2.0 m/s² -> throttle of 0.8
            throttle = float(np.clip(a_cmd / 2.5, 0.0, 1.0))
            brake = 0.0
        else:
            # Negative acceleration -> brake
            # Scale: a_cmd of -3.0 m/s² -> brake of 1.0
            throttle = 0.0
            brake = float(np.clip(-a_cmd / 3.0, 0.0, 1.0))
        
        return throttle, brake

    def _compute_reward(self, gap: float, v_av: float, v_hdv: float, 
                       rel_speed: float, headway: float, a_rl: float) -> float:
        """
        Compute one-step reward for the RL agent.
        
        Encourages:
        1. Maintaining headway near 1.8 seconds
        2. Smooth acceleration (small a_rl)
        3. Safety (avoiding collisions)
        4. Progress (not falling too far behind)
        """
        r = 0.0

        # 1) Headway tracking: penalize squared error around desired headway
        e_h = headway - self.h_des
        headway_term = -self.w_h * (e_h / self.h_band) ** 2
        r += headway_term

        # 2) Comfort: penalize large acceleration commands
        comfort_term = -self.w_a * (a_rl / self.a_ref) ** 2
        r += comfort_term

        # 3) Safety: big penalty if gap too small
        if gap < self.d_safe_min:
            r += -self.w_safe

        # 4) Progress: penalize if far behind while lead vehicle is moving
        if v_hdv > 0.5 and gap > self.d_far:
            r += -self.w_far * ((gap - self.d_far) / self.d_far)

        return float(r)

    def _check_collision(self, gap: float) -> bool:
        """
        Simple collision check: gap below 1 meter is considered collision.
        (You can make this more sophisticated later.)
        """
        return gap < 1.0

    def render(self, mode="human") -> None:
        """Optional: update spectator camera (nice for visualization)."""
        if self.ego is None:
            return
        
        av_tf = self.ego.get_transform()
        fwd = av_tf.get_forward_vector()
        chase_loc = av_tf.location - fwd * 8.0 + carla.Location(z=5.0)
        chase_rot = carla.Rotation(pitch=-15, yaw=av_tf.rotation.yaw)
        self.spectator.set_transform(carla.Transform(chase_loc, chase_rot))

    def close(self) -> None:
        """Cleanup at environment close."""
        self._destroy_actors()
