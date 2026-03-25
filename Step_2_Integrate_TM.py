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

    # Asynchronous mode for now
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    # Get spawn points
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    # Blueprints
    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')

    # Spawn AV only
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
    num_npcs = 25  # adjust as desired

    random.shuffle(spawn_points)
    for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None and npc.id != av.id:
            npc.set_autopilot(True, tm_port)
            npc_vehicles.append(npc)

    print(f"Spawned {len(npc_vehicles)} NPC vehicles.")

    # Adaptive longitudinal controller
    controller = AdaptiveController(
        time_headway=3.0,
        standstill_gap=5.0,
        alpha=1.0,
        dt=0.1
    )

    # Spectator view
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    # Logging setup
    log_filename = "av_longitudinal_log_multi.csv"
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
            "time_headway"
        ])

        print("Starting simulation in mixed traffic...")
        v_des = 12.0  # fallback desired speed (m/s)

        for i in range(1, 2001):
            av_state = get_vehicle_state(av)
            v = math.hypot(av_state['vx'], av_state['vy'])

            # --- Find nearest leader in front ---
            actors = world.get_actors().filter('*vehicle*')

            av_tf = av.get_transform()
            av_loc = av_tf.location
            av_fwd = av_tf.get_forward_vector()
            av_fwd_xy = np.array([av_fwd.x, av_fwd.y])

            lead_actor = None
            lead_dist = float('inf')

            for veh in actors:
                if veh.id == av.id:
                    continue
                tf_v = veh.get_transform()
                loc_v = tf_v.location
                rel = loc_v - av_loc
                rel_xy = np.array([rel.x, rel.y])

                # Only consider vehicles in front
                if np.dot(rel_xy, av_fwd_xy) <= 0:
                    continue

                dist = np.linalg.norm(rel_xy)
                if dist < lead_dist:
                    lead_dist = dist
                    lead_actor = veh

            hdv_state = None
            if lead_actor is not None:
                tf_h = lead_actor.get_transform()
                vel_h = lead_actor.get_velocity()
                hdv_state = {
                    'x': tf_h.location.x,
                    'y': tf_h.location.y,
                    'vx': vel_h.x,
                    'vy': vel_h.y,
                    'yaw': math.radians(tf_h.rotation.yaw)
                }

            # --- ACC / speed control ---
            nominal_acc = 1.0

            if hdv_state is not None:
                a_cmd = controller.compute_control(av_state, hdv_state, nominal_acc)
            else:
                # fallback: speed control toward v_des
                k_p = 0.5
                a_cmd = k_p * (v_des - v)

            # Map acceleration to throttle / brake
            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)

            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            # Apply control (for now, steer = 0.0; later can reuse TM steering)
            av.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=0.0,
                manual_gear_shift=False
            ))

            # Update spectator to chase view
            av_tf = av.get_transform()
            chase_loc = av_tf.location - av_tf.get_forward_vector() * 8 + carla.Location(z=5)
            chase_rot = carla.Rotation(pitch=-15, yaw=av_tf.rotation.yaw)
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            # --- Logging leader info and headway ---
            lead_id = -1
            lead_x = lead_y = lead_speed = float('nan')
            dist_headway = time_headway = float('nan')

            if hdv_state is not None:
                lead_id = lead_actor.id
                lead_x = hdv_state['x']
                lead_y = hdv_state['y']
                lead_speed = math.hypot(hdv_state['vx'], hdv_state['vy'])
                dist_headway = lead_dist
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
                time_headway
            ])

            print(
                f"Iter {i:04d}: "
                f"v={v:.2f} m/s, a_cmd={a_cmd:.2f}, "
                f"throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}, "
                f"lead_id={lead_id}, dist={dist_headway:.2f} m, "
                f"t_headway={time_headway:.2f} s"
            )

            time.sleep(controller.dt)

    print(f"Simulation complete. Log saved to {log_filename}")


if __name__ == '__main__':
    main()
