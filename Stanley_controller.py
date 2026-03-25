import math
import numpy as np


class StanleyController:
    def __init__(self,
                 k_cte=0.3,          # reduced cross-track gain
                 k_soft=1.0,         # softening for low speed
                 max_steer_rate=0.1, # tighter steering rate limit [rad/step]
                 lpf_alpha=0.3,      # stronger low-pass filtering
                 wheel_base=2.8,     # approx Tesla 3 wheelbase
                 steer_scale=0.7):   # global scaling of steering for CARLA
        self.k_cte = k_cte
        self.k_soft = k_soft
        self.max_steer_rate = max_steer_rate
        self.lpf_alpha = lpf_alpha
        self.wheel_base = wheel_base
        self.steer_scale = steer_scale

        self.prev_steer = 0.0
        self.filtered_steer = 0.0

        # For logging / debugging
        self.last_cte = 0.0

    def rate_limit(self, new_val, prev_val):
        delta = new_val - prev_val
        if delta > self.max_steer_rate:
            return prev_val + self.max_steer_rate
        if delta < -self.max_steer_rate:
            return prev_val - self.max_steer_rate
        return new_val

    def compute_control(self, av, v, target_loc, path_yaw):
        """
        av         : ego CARLA vehicle actor
        v          : ego speed [m/s]
        target_loc : carla.Location of look-ahead lane waypoint
        path_yaw   : heading of lane at that waypoint [rad]
        returns steer command in [-1, 1]
        """
        tf = av.get_transform()
        x = tf.location.x
        y = tf.location.y
        yaw = math.radians(tf.rotation.yaw)

        # 1) Heading error (wrap to [-pi, pi])
        heading_error = path_yaw - yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # 2) Front axle position
        fx = x + self.wheel_base * math.cos(yaw)
        fy = y + self.wheel_base * math.sin(yaw)

        # 3) Cross-track error relative to lane center
        dx = fx - target_loc.x
        dy = fy - target_loc.y

        # Path normal pointing to the left of lane
        path_normal_x = -math.sin(path_yaw)
        path_normal_y =  math.cos(path_yaw)

        cte = -(dx * path_normal_x + dy * path_normal_y)
        # If car consistently turns the wrong way, flip:
        # cte = -cte

        self.last_cte = cte

        # 4) Raw Stanley steering
        v_safe = max(v, 0.1)
        cte_term = math.atan2(self.k_cte * cte, self.k_soft + v_safe)

        # Limit how much CTE can influence steering
        max_cte_term = 0.4  # rad
        cte_term = float(np.clip(cte_term, -max_cte_term, max_cte_term))

        steer_raw = heading_error + cte_term

        # 5) Rate limit steering change
        steer_rl = self.rate_limit(steer_raw, self.prev_steer)

        # 6) Low-pass filter for smoothness
        steer_filt = (self.lpf_alpha * steer_rl +
                      (1.0 - self.lpf_alpha) * self.filtered_steer)

        self.prev_steer = steer_filt
        self.filtered_steer = steer_filt

        # 7) Scale and clip to CARLA range
        steer_cmd = self.steer_scale * steer_filt
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

        return steer_cmd
