import time
import csv
import math

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

    # Optional: leave asynchronous mode for now
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
    log_filename = "av_longitudinal_log.csv"
    with open(log_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "x", "y", "yaw",
            "speed",
            "a_cmd",
            "throttle_cmd",
            "brake_cmd"
        ])

        print("Starting simulation...")
        v_des = 12.0  # m/s desired speed

        for i in range(1, 1501):
            av_state = get_vehicle_state(av)

            # Current speed
            v = math.hypot(av_state['vx'], av_state['vy'])

            # If you added a dedicated speed control method:
            # a_cmd = controller.compute_speed_control(av_state, v_des)
            # Temporary simple proportional controller if that method does not exist:
            k_p = 0.5
            a_cmd = k_p * (v_des - v)

            # Map acceleration to throttle / brake
            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)

            # Clip to [0, 1]
            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            # For now, keep steering at 0.0 (straight); later you can reuse TM steering
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

            # Log data
            writer.writerow([
                i,
                av_state['x'],
                av_state['y'],
                av_state['yaw'],
                v,
                a_cmd,
                throttle_cmd,
                brake_cmd
            ])

            print(
                f"Iter {i:03d}: "
                f"x={av_state['x']:.2f}, y={av_state['y']:.2f}, "
                f"yaw={av_state['yaw']:.3f}, v={v:.2f} m/s, "
                f"a_cmd={a_cmd:.2f}, throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}"
            )

            time.sleep(controller.dt)

    print(f"Simulation complete. Log saved to {log_filename}")


if __name__ == '__main__':
    main()
