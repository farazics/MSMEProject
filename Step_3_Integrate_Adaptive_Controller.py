import glob
import os
import time
import csv
import math
import random

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


def main():
    # Auto-increment run id 
    existing = glob.glob("CBF_Controller_run*.csv")
    if not existing:
        run_id = 1
    else:
        numbers = []
        for fname in existing:
            base = os.path.basename(fname)          # e.g. CBF_Controller_run3.csv
            digits = "".join(ch for ch in base if ch.isdigit())
            if digits:
                numbers.append(int(digits))
        run_id = max(numbers) + 1 if numbers else 1

    print(f"Starting RL run_id = {run_id}")
    client = None
    world = None
    tm = None
    original_settings = None

    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        return

    # Traffic Manager setup
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    original_settings = world.get_settings()

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True           # world waits for world.tick()
    settings.fixed_delta_seconds = 0.08         # match controller.dt
    world.apply_settings(settings)

    tm.set_synchronous_mode(True)

    # Get spawn points
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    # Blueprints
    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')

    # Spawn AV
    av = world.try_spawn_actor(av_bp, av_tf)
    if av is None:
        print("ERROR: Could not spawn AV.")
        return

    # Register AV with Traffic Manager (autopilot)
    av.set_autopilot(True, tm_port)

    # Spawn NPC vehicles under TM
    vehicle_bps = bp_lib.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    npc_vehicles = []
    num_npcs = 65  # start small to avoid jams

    random.shuffle(spawn_points)

    for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None and npc.id != av.id:
            npc.set_autopilot(True, tm_port)
            npc_vehicles.append(npc)

    print(f"Spawned {len(npc_vehicles)} NPC vehicles registered to TM.")

    # ---- Collision sensor setup ----
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

    # Adaptive longitudinal controller
    controller = AdaptiveController(
        time_headway=2.0,    # was 3.0 → match new default
        standstill_gap=6.0,  # was 5.0 → match new default
        alpha=0.5,           # was 1.0 → less aggressive
        dt=0.08
    )

    # Spectator view
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    # Initialize sync state with a first tick before the main loop
    world.tick()

    # Logging setup
    log_filename = f"CBF_Controller_run{run_id}.csv"
    with open(log_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "av_x", "av_y", "av_yaw",
            "av_speed",
            "a_cmd",
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

        print("Starting simulation with TM traffic + ACC...")
        v_des = 12.0  # free-flow desired speed
        nominal_acc = 1.0

        for i in range(1, 10001):
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

                # Vector from AV to this vehicle
                rel = loc_v - ego_loc
                dot = (
                    rel.x * ego_forward.x +
                    rel.y * ego_forward.y +
                    rel.z * ego_forward.z
                )

                # Must be in front
                if dot <= 0:
                    continue

                # Same lane (approx)
                veh_wp = world.get_map().get_waypoint(
                    loc_v, project_to_road=True
                )
                if (veh_wp.road_id != ego_wp.road_id or
                        veh_wp.lane_id != ego_wp.lane_id):
                    continue

                dist = math.sqrt(rel.x**2 + rel.y**2 + rel.z**2)
                if dist < leader_dist and dist > 1.0:
                    leader_dist = dist
                    leader = veh

            # Build hdv_state if we found leader
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

            # --- ACC / speed control ---
            if hdv_state is not None:
                a_cmd = controller.compute_control(
                    av_state, hdv_state, nominal_acc
                )
            else:
                # Free road: simple speed control toward v_des
                k_p = 0.3              # was 0.5 → gentler proportional gain
                a_cmd = k_p * (v_des - v)
                a_cmd = float(np.clip(a_cmd, -3.0, 1.96))  # Free-road startup acceleration cap

            # Map acceleration to throttle / brake
            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)

            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            # Apply control (for now, steer = 0.0; TM is mostly for routing)
            av.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=0.0,
                manual_gear_shift=False
            ))

            # Update spectator to chase view
            av_tf_curr = av.get_transform()
            chase_loc = (av_tf_curr.location -
                         av_tf_curr.get_forward_vector() * 8 +
                         carla.Location(z=5))
            chase_rot = carla.Rotation(
                pitch=-15, yaw=av_tf_curr.rotation.yaw
            )
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            # --- Logging ---
            lead_id = -1
            lead_x = lead_y = lead_speed = float('nan')
            dist_headway = time_headway = float('nan')

            if hdv_state is not None:
                lead_id = leader.id
                lead_x = hdv_state['x']
                lead_y = hdv_state['y']
                lead_speed = math.hypot(
                    hdv_state['vx'], hdv_state['vy']
                )
                dist_headway = leader_dist
                if v > 0.1:
                    time_headway = dist_headway / v

            writer.writerow([
                i,
                av_state['x'], av_state['y'], av_state['yaw'],
                v,
                a_cmd,
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

            print(
                f"Iter {i:04d}: "
                f"v={v:.2f} m/s, a_cmd={a_cmd:.2f}, "
                f"throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}, "
                f"lead_id={lead_id}, dist={dist_headway:.2f} m, "
                f"t_headway={time_headway:.2f} s"
            )

            world.tick()   # advance the simulation one step

    print(f"Simulation complete. Log saved to {log_filename}")



if __name__ == '__main__':
    main()
