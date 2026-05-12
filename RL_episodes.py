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
                    self.q = np.zeros(shape, dtype=np.float32)
            except FileNotFoundError:
                print(f"No existing Q-table at {q_init_path}, starting from zeros.")
                self.q = np.zeros(shape, dtype=np.float32)
        else:
            self.q = np.zeros(shape, dtype=np.float32)

    def save(self, path):
        np.save(path, self.q)
        print(f"Saved Q-table to {path}")

    def _bin_index(self, value, edges):
        for i in range(len(edges) - 1):
            if edges[i] <= value < edges[i + 1]:
                return i
        return len(edges) - 2

    def state_to_index(self, v, d, dv):
        v_i = self._bin_index(v, self.v_bins)
        d_i = self._bin_index(d, self.d_bins)
        dv_i = self._bin_index(dv, self.dv_bins)
        return (v_i, d_i, dv_i)

    def select_action(self, state_idx):
        if random.random() < self.eps:
            a_idx = random.randrange(len(self.actions))
        else:
            q_vals = self.q[state_idx]
            a_idx = int(np.argmax(q_vals))
        return a_idx, self.actions[a_idx]

    def update(self, s_idx, a_idx, reward, s_next_idx, done):
        q_sa = self.q[s_idx][a_idx]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[s_next_idx])
        self.q[s_idx][a_idx] = q_sa + self.alpha * (target - q_sa)
        self.eps = max(self.eps_end, self.eps - self.eps_decay)


def main():
    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        return

    actors_to_destroy = []

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return
    av_tf = spawns[0]

    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle_bps = bp_lib.filter('*vehicle*')

    av = world.try_spawn_actor(av_bp, av_tf)
    if av is None:
        print("ERROR: Could not spawn AV.")
        return
    av.set_autopilot(True, tm_port)
    actors_to_destroy.append(av)

    spawn_points = world.get_map().get_spawn_points()
    npc_vehicles = []
    num_npcs = 35
    random.shuffle(spawn_points)
    for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None and npc.id != av.id:
            npc.set_autopilot(True, tm_port)
            npc_vehicles.append(npc)
    print(f"Spawned {len(npc_vehicles)} NPC vehicles registered to TM.")
    actors_to_destroy.extend(npc_vehicles)

    # --- Collision sensor: longitudinal count + lateral termination flag ---
    collision_history = {
        "longitudinal_collision_count": 0,
        "lateral_collision_detected": False
    }

    def on_collision(event):
        ego_fwd = av.get_transform().get_forward_vector()
        impulse = event.normal_impulse
        long_component = abs(
            impulse.x * ego_fwd.x +
            impulse.y * ego_fwd.y +
            impulse.z * ego_fwd.z
        )
        lat_component = math.sqrt(
            (impulse.x - long_component * ego_fwd.x) ** 2 +
            (impulse.y - long_component * ego_fwd.y) ** 2
        )
        if long_component >= lat_component:
            collision_history["longitudinal_collision_count"] += 1
        else:
            collision_history["lateral_collision_detected"] = True
            print("LATERAL COLLISION DETECTED — AV left the path. Terminating run.")

    col_bp = bp_lib.find('sensor.other.collision')
    col_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
    collision_sensor = world.try_spawn_actor(col_bp, col_tf, attach_to=av)
    if collision_sensor is None:
        print("WARNING: Could not spawn collision sensor.")
    else:
        collision_sensor.listen(on_collision)
        actors_to_destroy.append(collision_sensor)

    # Adaptive longitudinal controller — time_headway = 1.5 s
    controller = AdaptiveController(
        time_headway=1.5,
        standstill_gap=5.0,
        alpha=1.0,
        dt=0.1
    )

    q_table_path = "q_table.npy"
    rl_agent = DiscreteRLAgent(q_init_path=q_table_path)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    log_filename = get_next_run_filename("RL_run")
    print(f"Logging to: {log_filename}")

    steps_total = 10000

    COMFORT_COLUMNS = [
        "step",
        "av_speed",
        "a_acc",
        "a_rl",
        "a_cmd_total",
        "jerk",
        "lead_speed",
        "distance_headway",
        "time_headway",
        "longitudinal_collision_count"
    ]

    v_des = 12.0
    nominal_acc = 1.0
    prev_a_cmd_total = 0.0
    collision_count_prev = 0
    terminated_early = False

    # Safe defaults so terminal Q-update never references unbound variables
    s_idx = (0, 0, 0)
    a_idx = 0

    with open(log_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COMFORT_COLUMNS)
        print("Starting simulation: RL + ACC with TM traffic...")

        for step in range(1, steps_total + 1):
            # Terminate if AV left the path (lateral collision)
            if collision_history["lateral_collision_detected"]:
                print(f"Run terminated at step {step} due to lateral collision (AV off-path).")
                rl_agent.update(s_idx, a_idx, -100.0, s_idx, done=True)
                terminated_early = True
                break

            av_state = get_vehicle_state(av)
            v = math.hypot(av_state['vx'], av_state['vy'])

            ego_tf = av.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()
            ego_wp = world.get_map().get_waypoint(ego_loc, project_to_road=True)

            vehicles = world.get_actors().filter('vehicle.*')
            leader = None
            leader_dist = float('inf')

            for veh in vehicles:
                if veh.id == av.id:
                    continue
                tf_v = veh.get_transform()
                loc_v = tf_v.location
                rel = loc_v - ego_loc
                dot = (
                    rel.x * ego_forward.x +
                    rel.y * ego_forward.y +
                    rel.z * ego_forward.z
                )
                if dot <= 0:
                    continue
                veh_wp = world.get_map().get_waypoint(loc_v, project_to_road=True)
                if (veh_wp.road_id != ego_wp.road_id or
                        veh_wp.lane_id != ego_wp.lane_id):
                    continue
                dist = math.sqrt(rel.x ** 2 + rel.y ** 2 + rel.z ** 2)
                if dist < leader_dist and dist > 1.0:
                    leader_dist = dist
                    leader = veh

            hdv_state = None
            if leader is not None:
                tf_l = leader.get_transform()
                vel_l = leader.get_velocity()
                hdv_state = {
                    'x': tf_l.location.x,
                    'y': tf_l.location.y,
                    'vx': vel_l.x,
                    'vy': vel_l.y,
                    'yaw': math.radians(tf_l.rotation.yaw)
                }

            if hdv_state is not None:
                a_acc = controller.compute_control(av_state, hdv_state, nominal_acc)
            else:
                k_p = 0.5
                a_acc = k_p * (v_des - v)

            a_acc = float(np.clip(a_acc, -4.0, 2.0))

            if hdv_state is not None:
                v_lead = math.hypot(hdv_state['vx'], hdv_state['vy'])
                dist_headway = leader_dist
            else:
                v_lead = v_des
                dist_headway = 80.0

            dv = v_lead - v
            s_idx = rl_agent.state_to_index(v, dist_headway, dv)
            a_idx, a_rl = rl_agent.select_action(s_idx)

            a_cmd_total = float(np.clip(a_acc + a_rl, -4.0, 2.0))
            jerk = (a_cmd_total - prev_a_cmd_total) / controller.dt

            reward = -0.1 * (a_cmd_total ** 2) - 0.05 * (jerk ** 2)

            time_headway = float('nan')
            if v > 0.1:
                time_headway = dist_headway / v
            if not math.isnan(time_headway) and time_headway < 1.5:
                reward -= 5.0

            collision_count_now = collision_history["longitudinal_collision_count"]
            if collision_count_now > collision_count_prev:
                collisions_this_step = collision_count_now - collision_count_prev
                reward -= 50.0 * collisions_this_step
                collision_count_prev = collision_count_now

            s_next_idx = rl_agent.state_to_index(v, dist_headway, dv)
            rl_agent.update(s_idx, a_idx, reward, s_next_idx, False)
            prev_a_cmd_total = a_cmd_total

            throttle_cmd = float(np.clip(max(a_cmd_total, 0.0), 0.0, 1.0))
            brake_cmd = float(np.clip(max(-a_cmd_total, 0.0), 0.0, 1.0))

            av.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=0.0,
                manual_gear_shift=False
            ))

            av_tf_curr = av.get_transform()
            chase_loc = (av_tf_curr.location -
                         av_tf_curr.get_forward_vector() * 8 +
                         carla.Location(z=5))
            chase_rot = carla.Rotation(pitch=-15, yaw=av_tf_curr.rotation.yaw)
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            lead_speed = math.hypot(hdv_state['vx'], hdv_state['vy']) if hdv_state else float('nan')

            writer.writerow([
                step,
                round(v, 4),
                round(a_acc, 4),
                round(a_rl, 4),
                round(a_cmd_total, 4),
                round(jerk, 4),
                round(lead_speed, 4) if not math.isnan(lead_speed) else lead_speed,
                round(dist_headway, 4),
                round(time_headway, 4) if not math.isnan(time_headway) else time_headway,
                collision_history["longitudinal_collision_count"]
            ])

            print(
                f"Step {step:05d}: v={v:.2f} m/s, a_acc={a_acc:.2f}, "
                f"a_rl={a_rl:.2f}, a_total={a_cmd_total:.2f}, "
                f"jerk={jerk:.2f} m/s3, TH={time_headway:.2f} s, "
                f"lon_col={collision_history['longitudinal_collision_count']}"
            )

            time.sleep(controller.dt)

    rl_agent.save(q_table_path)

    if terminated_early:
        print(f"Run ended early — lateral collision (AV off-path). Q-table saved. Data saved to {log_filename}")
    else:
        print(f"Simulation complete. Log saved to {log_filename}")

    for a in actors_to_destroy:
        try:
            a.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    main()
