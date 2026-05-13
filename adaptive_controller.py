import numpy as np
import cvxpy as cp


class AdaptiveController:
    """
    Adaptive Cruise Control using Control Barrier Functions (CBF) with QP optimization.
    Ensures safe following distance while tracking a nominal acceleration.
    """

    def __init__(self,
                 time_headway: float = 2.0,      # was 1.8 → more gap = gentler following
                 standstill_gap: float = 5.0,    # 5 m at standstill
                 alpha: float = 1.0,
                 dt: float = 0.1,
                 dsafe: float = 25.0,
                 h_threshold: float = 0.1,
                 max_acceleration: float = 1.96, # was 3.0 → caps at Tesla 0.20 g
                 min_acceleration: float = -3.0): # was -5.0 → caps braking at 0.30 g
        # Safety and control parameters
        self.Th = time_headway          # Time headway (seconds)
        self.d0 = standstill_gap        # Minimum standstill gap (meters)
        self.alpha = alpha              # CBF gain
        self.dt = dt
        self.dsafe = dsafe
        self.h_threshold = h_threshold
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration

        # CVXPY optimization setup
        self.u = cp.Variable()          # acceleration command
        self.h_value = cp.Parameter()   # CBF value h
        self.rel_speed = cp.Parameter() # v_hdv - v_av
        self.u_nominal = cp.Parameter() # desired (nominal) acceleration

        # Objective: minimize deviation from nominal acceleration
        objective = cp.Minimize(0.5 * cp.square(self.u - self.u_nominal))

        # Control Barrier Function (CBF) constraint:
        #   rel_speed - Th * u + alpha * h >= 0
        cbf_constraint = self.rel_speed - self.Th * self.u + self.alpha * self.h_value >= 0

        # Acceleration bounds
        bounds = [
            self.u >= self.min_acceleration,
            self.u <= self.max_acceleration
        ]

        # Define QP problem
        self.problem = cp.Problem(objective, [cbf_constraint, *bounds])

        # Smoothing and filtering state
        self.accel_prev = 0.0
        self.filtered_accel = 0.0
        self.max_accel_change = 1.0 * dt  # rate limit per step was 2.0 → jerk limiter: max 1.0 m/s³
        self.lpf_alpha = 0.3              # low-pass filter coefficient was 0.5 → stronger smoothing filter

    def blend_control(self, nominal: float, constrained: float, h: float, threshold: float = 0.3) -> float:
        """
        Smoothly blend nominal and constrained controls based on safety margin.
        - h >= threshold: fully nominal control
        - h <= 0       : fully constrained control
        - in-between   : quadratic blend
        """
        if h >= threshold:
            return nominal
        elif h <= 0:
            return constrained
        else:
            alpha = (h / threshold) ** 2
            return alpha * nominal + (1 - alpha) * constrained

    def rate_limit(self, new_val: float, prev_val: float, max_delta: float) -> float:
        """
        Apply rate limiting to prevent abrupt acceleration changes.
        """
        delta = new_val - prev_val
        if delta > max_delta:
            return prev_val + max_delta
        elif delta < -max_delta:
            return prev_val - max_delta
        else:
            return new_val

    def compute_control(self,
                        av_state: dict,
                        hdv_state: dict,
                        nominal_acceleration: float) -> float:
        # Relative position and gap
        dx = hdv_state['x'] - av_state['x']
        dy = hdv_state['y'] - av_state['y']
        gap = np.hypot(dx, dy)

        # Speeds
        v_av = np.hypot(av_state['vx'], av_state['vy'])
        v_hdv = np.hypot(hdv_state['vx'], hdv_state['vy'])

        # Control Barrier Function value:
        #   h = gap - v_av * Th - d0
        h = gap - v_av * self.Th - self.d0

        # Relative speed (HDV - AV)
        rel_speed = v_hdv - v_av

        # ---------------- STANDSTILL / HOLD LOGIC ----------------
        vel_tol = 0.3           # m/s, below this = effectively stopped
        hold_gap_thresh = 12.0  # m, do not creep closer than this

        if (v_av < vel_tol) and (v_hdv < vel_tol) and (gap <= hold_gap_thresh):
            # Both effectively stopped and close: hold position
            accel_final = 0.0
            self.accel_prev = accel_final
            self.filtered_accel = accel_final
            return accel_final
        # ---------------------------------------------------------

        # Update QP parameters
        self.h_value.value = h
        self.rel_speed.value = rel_speed

        # ---------------- EMERGENCY COLLISION AVOIDANCE ----------------
        emergency_distance = 5.5        # m
        closing_speed_margin = 2.0      # m/s

        if (gap < emergency_distance) and (v_av > v_hdv + closing_speed_margin):
            # Real collision risk: hard brake
            accel_final = self.min_acceleration  # -5 m/s²
            self.accel_prev = accel_final
            self.filtered_accel = accel_final
            return accel_final
        # ---------------------------------------------------------------

        # ---------------- FOLLOWING / SPEED-MATCHING LOGIC ----------------
        # Base nominal behavior: desire to accelerate at nominal_acceleration
        u_nom = np.clip(nominal_acceleration, self.min_acceleration, self.max_acceleration)

        # Distance zones
        follow_far = 20.0  # m
        follow_near = 12.0 # m

        if h < 0:
            # Inside desired headway; modify behavior based on gap

            if gap > follow_far:
                # Zone A: far but inside headway numerically; still allow nominal accel
                # QP will keep it safe.
                pass

            elif follow_near < gap <= follow_far:
                # Zone B: follow region ~12–20m
                # Gentle speed matching: do NOT drive AV to 0 while HDV is moving.
                k_v = 0.3  # smaller gain
                u_follow = k_v * (v_hdv - v_av)  # negative if AV faster

                # Limit braking strength in this zone for comfort:
                # - do not brake harder than -0.5 m/s² when gap > 12m
                u_follow = max(u_follow, -0.3) # was -0.5 → gentler mid-range braking

                # Also, avoid slowing below HDV speed here:
                # if AV is already slower or equal, don't brake more.
                if v_av <= v_hdv:
                    u_follow = max(u_follow, 0.0)

                u_nom = np.clip(u_follow, self.min_acceleration, self.max_acceleration)

            else:
                # Zone C: gap <= follow_near (closer than ~12m, but not in emergency zone)
                # Stronger speed matching is allowed, but still not extreme.
                k_v = 0.5
                u_follow = k_v * (v_hdv - v_av)

                # In this close region, allow up to -2 m/s² for a stronger but still smooth brake
                u_follow = max(u_follow, -1.5) # was -2.0 → still safe but less abrupt
                u_nom = np.clip(u_follow, self.min_acceleration, self.max_acceleration)
        # -----------------------------------------------------------------

        # Set nominal for QP
        self.u_nominal.value = u_nom

        # Solve QP
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=False)
            if self.problem.status == cp.OPTIMAL:
                u_constrained = float(self.u.value)
            else:
                u_constrained = self.min_acceleration
        except Exception:
            u_constrained = self.min_acceleration

        # Blend nominal and constrained accelerations
        accel_blended = self.blend_control(u_nom, u_constrained, h, threshold=0.0)

        # Rate limit
        accel_limited = self.rate_limit(accel_blended, self.accel_prev, self.max_accel_change)

        # Low-pass filter
        accel_filtered = self.lpf_alpha * accel_limited + (1 - self.lpf_alpha) * self.filtered_accel

        # Store state
        self.accel_prev = accel_filtered
        self.filtered_accel = accel_filtered

        return accel_filtered
