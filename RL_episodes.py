import time
import csv
import math
import random
import numpy as np
import carla
import glob
import os

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


# ===================== RL Agent =====================
class DiscreteRLAgent:
    def __init__(self,
                 v_bins=(0.0, 5.0, 10.0, 15.0, 25.0),
                 d_bins=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0),
                 dv_bins=(-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0),
                 actions=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
                 alpha=0.05,
                 gamma=0.99,
                 eps_start=0.5,
                 eps_end=0.05,
                 eps_decay=5e-4,
                 q_init_path=None):
        # Discretization
        self.v_bins = v_bins
        self.d_bins = d_bins
        self.dv_bins = dv_bins
        self.actions = actions

        # RL params
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Q-table shape
        shape = (len(v_bins) - 1,
                 len(d_bins) - 1,
                 len(dv_bins) - 1,
                 len(actions))

        # Initialize or load Q-table
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
        # returns index [0, len(edges)-2]
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
        # epsilon-greedy
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

        # decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)


def main():
    # Auto-increment run id based on existing RL_long_run*.csv files
    existing = glob.glob("RL_long_run*.csv")
    if not existing:
        run_id = 1
    else:
        numbers = []
        for fname in existing:
            base = os.path.basename(fname)          # e.g. RL_long_run3.csv
            digits = "".join(ch for ch in base if ch.isdigit())
            if digits:
                numbers.append(int(digits))
        run_id = max(numbers) + 1 if numbers else 1

    print(f"Starting RL run_id = {run_id}")
    
    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        return

    # Track actors we spawn so we can destroy them at the end
    actors_to_destroy = []

    # Traffic Manager setup
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1         # match your controller dt
    world.apply_settings(settings)

    tm.set_synchronous_mode(True)              # TM runs in lockstep with world

    # Get spawn points
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return
    av_tf = spawns[0]

    # Blueprints
    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle_bps = bp_lib.filter('*vehicle*')

    # Spawn AV
    av = world.try_spawn_actor(av_bp, av_tf)
    if av is None:
        print("ERROR: Could not spawn AV.")
        return
    av.set_autopilot(True, tm_port)
    actors_to_destroy.append(av)

    # Spawn NPC vehicles
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

    # Collision sensor setup
    collision_history = {"collision_count": 0}

    def on_collision(event):
        collision_history["collision_count"] += 1

    col_bp = bp_lib.find('sensor.other.collision')
    col_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
    collision_sensor = world.try_spawn_actor(col_bp, col_tf, attach_to=av)
    if collision_sensor is None:
        print("WARNING: Could not spawn collision sensor.")
    else:
        collision_sensor.listen(on_collision)
        actors_to_destroy.append(collision_sensor)

    # Adaptive longitudinal controller
    controller = AdaptiveController(
        time_headway=3.0,
        standstill_gap=5.0,
        alpha=1.0,
        dt=0.1
    )

    # RL agent with persistent Q-table
    q_table_path = "q_table.npy"
    rl_agent = DiscreteRLAgent(q_init_path=q_table_path)

    # Spectator view
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    # Logging setup: one long run
    log_filename = f"RL_long_run{run_id}.csv"
    steps_total = 10000  # e.g. 10k steps

    with open(log_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "av_x", "av_y", "av_yaw",
            "av_speed",
            "a_acc",        # ACC acceleration only
            "a_rl",         # RL correction
            "a_cmd_total",  # a_acc + a_rl
            "jerk",
            "throttle_cmd",
            "brake_cmd",
            "lead_id",
            "lead_x",
            "lead_y",
            "lead_speed",
            "distance_headway",
            "time_headway",
            "collision_count"
        ])

        print("Starting long simulation with TM traffic + ACC + RL...")

        v_des = 12.0
        nominal_acc = 1.0
        prev_a_cmd_total = 0.0
        collision_count_prev = 0  # for collision-based reward

        for step in range(1, steps_total + 1):
            # Ego state
            av_state = get_vehicle_state(av)
            v = math.hypot(av_state['vx'], av_state['vy'])

            # --- Find closest leader ahead in same lane ---
            ego_tf = av.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()
            ego_wp = world.get_map().get_waypoint(
                ego_loc, project_to_road=True
            )

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

                veh_wp = world.get_map().get_waypoint(
                    loc_v, project_to_road=True
                )
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

            # ACC baseline
            if hdv_state is not None:
                a_acc = controller.compute_control(
                    av_state, hdv_state, nominal_acc
                )
            else:
                k_p = 0.5
                a_acc = k_p * (v_des - v)

            a_acc = float(np.clip(a_acc, -4.0, 2.0))

            # RL state
            if hdv_state is not None:
                v_lead = math.hypot(hdv_state['vx'], hdv_state['vy'])
                dist_headway = leader_dist
            else:
                v_lead = v_des
                dist_headway = 80.0

            dv = v_lead - v
            s_idx = rl_agent.state_to_index(v, dist_headway, dv)

            # RL action
            a_idx, a_rl = rl_agent.select_action(s_idx)

            # Combine ACC + RL
            a_cmd_total = a_acc + a_rl
            a_cmd_total = float(np.clip(a_cmd_total, -4.0, 2.0))

            # Reward: comfort + speed tracking + safety + collisions

            # Comfort penalties
            jerk = (a_cmd_total - prev_a_cmd_total) / controller.dt
            reward = -0.1 * (a_cmd_total ** 2) - 0.05 * (jerk ** 2)
            
            # Speed tracking: small penalty for being far from desired speed
            speed_error = v_des - v
            reward -= 0.01 * (speed_error ** 2)
            
            # Time headway / safety
            time_headway = float('nan')
            if v > 0.1:
                time_headway = dist_headway / v
            if time_headway < 1.0:
                reward -= 5.0  # unsafe headway penalty
            
            # Collision penalty: new collisions since last step
            collision_count_now = collision_history["collision_count"]
            if collision_count_now > collision_count_prev:
                collisions_this_step = collision_count_now - collision_count_prev
                reward -= 100.0 * collisions_this_step  # stronger penalty per collision
            collision_count_prev = collision_count_now

            # RL update
            s_next_idx = rl_agent.state_to_index(v, dist_headway, dv)
            done = False
            rl_agent.update(s_idx, a_idx, reward, s_next_idx, done)
            prev_a_cmd_total = a_cmd_total

            # Map to throttle/brake
            throttle_cmd = max(a_cmd_total, 0.0)
            brake_cmd = max(-a_cmd_total, 0.0)
            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            av.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=0.0,
                manual_gear_shift=False
            ))

            # Spectator chase view
            av_tf_curr = av.get_transform()
            chase_loc = (av_tf_curr.location -
                         av_tf_curr.get_forward_vector() * 8 +
                         carla.Location(z=5))
            chase_rot = carla.Rotation(
                pitch=-15, yaw=av_tf_curr.rotation.yaw
            )
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            # Logging
            lead_id = -1
            lead_x = lead_y = lead_speed = float('nan')
            if hdv_state is not None:
                lead_id = leader.id
                lead_x = hdv_state['x']
                lead_y = hdv_state['y']
                lead_speed = math.hypot(
                    hdv_state['vx'], hdv_state['vy']
                )

            writer.writerow([
                step,
                av_state['x'], av_state['y'], av_state['yaw'],
                v,
                a_acc,
                a_rl,
                a_cmd_total,
                jerk,
                throttle_cmd,
                brake_cmd,
                lead_id,
                lead_x,
                lead_y,
                lead_speed,
                dist_headway,
                time_headway,
                collision_history["collision_count"]
            ])

            world.tick()                # <-- advance simulation one step

    # Save Q-table after long run
    rl_agent.save(q_table_path)
    print(f"Simulation complete. Log saved to {log_filename}")

     # Also keep a copy as RL_long_latest.csv for quick inspection
    import shutil
    shutil.copyfile(log_filename, "RL_long_latest.csv")

    # Clean up actors (including sensor) to avoid warnings
    for a in actors_to_destroy:
        try:
            a.destroy()
        except Exception:
            pass

    # Restore async mode before exit
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)

if __name__ == '__main__':
    main()
