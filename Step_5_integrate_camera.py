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
    client = None
    world = None
    tm = None
    original_settings = None
    av = None
    npc_vehicles = []
    camera = None

    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")

        # Save original settings
        original_settings = world.get_settings()

        # --- Synchronous mode, near real-time ---
        settings = world.get_settings()
        settings.synchronous_mode = True        # client controls ticks
        settings.fixed_delta_seconds = 0.1      # 10 Hz, matches controller.dt
        world.apply_settings(settings)

        # Traffic Manager in sync mode
        tm = client.get_trafficmanager()
        tm_port = tm.get_port()
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        # --- Spawns and blueprints ---
        spawns = world.get_map().get_spawn_points()
        if not spawns:
            print("ERROR: No spawn points found in map.")
            return

        av_tf = spawns[0]

        bp_lib = world.get_blueprint_library()
        av_bp = bp_lib.find('vehicle.tesla.model3')

        # --- Spawn AV ---
        av = world.try_spawn_actor(av_bp, av_tf)
        if av is None:
            print("ERROR: Could not spawn AV.")
            return

        # Enable TM control for routing + baseline behavior
        av.set_autopilot(True, tm_port)

        # --- Attach RGB camera to AV ---
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.1')  # ~ controller.dt

        cam_loc = carla.Location(x=2.0, y=0.0, z=1.4)
        cam_rot = carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
        cam_tf = carla.Transform(cam_loc, cam_rot)

        camera = world.spawn_actor(camera_bp, cam_tf, attach_to=av)

        # Image output directory and mutable frame tracker
        img_dir = "adv_images"
        os.makedirs(img_dir, exist_ok=True)
        latest_cam_frame = {'frame_id': -1}

        def camera_callback(image):
            image.save_to_disk(os.path.join(img_dir, f"{image.frame}.png"))
            latest_cam_frame['frame_id'] = image.frame

        camera.listen(camera_callback)

        # --- Spawn NPCs under TM ---
        vehicle_bps = bp_lib.filter('*vehicle*')
        spawn_points = world.get_map().get_spawn_points()

        npc_vehicles = []
        num_npcs = 45

        random.shuffle(spawn_points)
        for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
            npc = world.try_spawn_actor(bp, sp)
            if npc is not None and npc.id != av.id:
                npc.set_autopilot(True, tm_port)
                npc_vehicles.append(npc)

        print(f"Spawned {len(npc_vehicles)} NPC vehicles.")

        # --- Controller and spectator ---
        controller = AdaptiveController(
            time_headway=3.0,
            standstill_gap=5.0,
            alpha=1.0,
            dt=0.1
        )

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            av_tf.location + carla.Location(z=30),
            carla.Rotation(pitch=-90)
        ))

        # --- Logging buffer ---
        log_rows = []
        log_filename = "adv_log.csv"

        print("Starting synchronous simulation (near real-time) with TM + ACC + camera...")

        v_des = 12.0
        nominal_acc = 1.0
        num_steps = 1500

        for i in range(1, num_steps + 1):
            # --------------------------------------------------------------
            # 1) Read ego state
            # --------------------------------------------------------------
            av_state = get_vehicle_state(av)
            v_av = math.hypot(av_state['vx'], av_state['vy'])

            ego_tf = av.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()
            ego_wp = world.get_map().get_waypoint(
                ego_loc, project_to_road=True
            )

            # --------------------------------------------------------------
            # 2) Find closest leader ahead in same lane
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # 3) ACC / speed control
            # --------------------------------------------------------------
            if hdv_state is not None:
                a_cmd = controller.compute_control(
                    av_state, hdv_state, nominal_acc
                )
            else:
                k_p = 0.5
                a_cmd = k_p * (v_des - v_av)

            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)

            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            # --------------------------------------------------------------
            # 4) Apply controls – keep TM steering
            # --------------------------------------------------------------
            tm_control = av.get_control()  # last applied control
            av.apply_control(carla.VehicleControl(
                throttle=throttle_cmd,
                brake=brake_cmd,
                steer=tm_control.steer,
                manual_gear_shift=False
            ))

            # Spectator chase view
            av_tf_curr = av.get_transform()
            chase_loc = (
                av_tf_curr.location -
                av_tf_curr.get_forward_vector() * 8 +
                carla.Location(z=5)
            )
            chase_rot = carla.Rotation(
                pitch=-15, yaw=av_tf_curr.rotation.yaw
            )
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            # --------------------------------------------------------------
            # 5) Tick world once and keep wall-time ~ sim-time
            # --------------------------------------------------------------
            world.tick()
            snapshot = world.get_snapshot()
            t = snapshot.timestamp.elapsed_seconds

            time.sleep(0.1)

            # --------------------------------------------------------------
            # 6) Log row (with camera frame)
            # --------------------------------------------------------------
            x_av = av_state['x']
            y_av = av_state['y']
            yaw_av = av_state['yaw']

            x_lead = float('nan')
            y_lead = float('nan')
            v_lead = float('nan')
            dist = float('nan')
            time_headway = float('nan')
            lead_id = -1

            if hdv_state is not None:
                x_lead = hdv_state['x']
                y_lead = hdv_state['y']
                v_lead = math.hypot(hdv_state['vx'], hdv_state['vy'])
                dist = leader_dist
                lead_id = leader.id
                if v_av > 0.1:
                    time_headway = dist / v_av

            cam_frame = latest_cam_frame['frame_id']

            row = [
                t,
                x_av, y_av, yaw_av,
                v_av,
                x_lead, y_lead, v_lead,
                dist, time_headway,
                a_cmd, throttle_cmd, brake_cmd,
                cam_frame
            ]
            log_rows.append(row)

            print(
                f"Iter {i:04d}: "
                f"t={t:.2f} s, v_av={v_av:.2f} m/s, a_cmd={a_cmd:.2f}, "
                f"throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}, "
                f"lead_id={lead_id}, dist={dist:.2f} m, "
                f"t_headway={time_headway:.2f} s, cam_frame={cam_frame}"
            )

        # --- Write log once at end ---
        with open(log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't',
                'x_av', 'y_av', 'yaw_av',
                'v_av',
                'x_lead', 'y_lead', 'v_lead',
                'dist', 'time_headway',
                'a_cmd', 'throttle_cmd', 'brake_cmd',
                'cam_frame'
            ])
            writer.writerows(log_rows)

        print(f"Simulation complete. Log saved to {log_filename}")
        print(f"Images saved to directory: {img_dir}")

        # --- Cleanup sensors/actors ---
        if camera is not None:
            camera.stop()
            camera.destroy()
        for npc in npc_vehicles:
            npc.destroy()
        if av is not None:
            av.destroy()

    finally:
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if tm is not None:
            tm.set_synchronous_mode(False)


if __name__ == '__main__':
    main()
