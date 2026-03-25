import time
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
        'yaw': np.deg2rad(tf.rotation.yaw)
    }


def main():
    try:
        # Connect to CARLA server
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
        # Optional: reset world
        # world = client.reload_world()
    except RuntimeError as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        print("Make sure CARLA server is running (CarlaUE4.exe or equivalent).")
        return
    except Exception as e:
        print(f"ERROR: Unexpected error connecting to CARLA: {e}")
        return

    # Get spawn points
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    # HDV spawns 25 m ahead along AV heading (more realistic urban-ish following)
    fwd = av_tf.get_forward_vector()
    hdv_location = av_tf.location + fwd * 25.0
    hdv_tf = carla.Transform(hdv_location, av_tf.rotation)

    # Retrieve blueprints
    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')
    hdv_bp = bp_lib.filter('vehicle.audi.tt')[0]

    # Spawn vehicles
    av = world.try_spawn_actor(av_bp, av_tf)
    hdv = world.try_spawn_actor(hdv_bp, hdv_tf)
    if av is None or hdv is None:
        print("ERROR: Could not spawn one or both vehicles.")
        return

    # 3-second rule: time_headway=3.0, standstill_gap=5.0
    controller = AdaptiveController(
        time_headway=3.0,
        standstill_gap=5.0,
        alpha=1.0,
        dt=0.1
    )

    # Set spectator view above AV's start
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    print("Starting simulation...")

    for i in range(1, 151):
        av_state = get_vehicle_state(av)
        hdv_state = get_vehicle_state(hdv)

        # HDV behavior: accelerate, short cruise, then brake to full stop
        if i <= 50:
            # Moderate acceleration from rest
            hdv.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0))
        elif 51 <= i <= 100:
            # Light cruise
            hdv.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0))
        else:
            # Smooth braking to a stop, then remain stopped
            hdv.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))

        # Nominal AV acceleration (what it would like to do if unconstrained)
        nominal_acc = 1.0

        # Get safe acceleration from controller
        a_cmd = controller.compute_control(av_state, hdv_state, nominal_acc)

        # Convert acceleration to throttle and brake (simple map)
        throttle_cmd = max(a_cmd, 0.0)
        brake_cmd = max(-a_cmd, 0.0)

        # Apply to AV
        av.apply_control(carla.VehicleControl(
            throttle=throttle_cmd,
            brake=brake_cmd,
            steer=0.0
        ))

        # Update spectator to chase-view behind AV
        av_tf = av.get_transform()
        chase_loc = av_tf.location - av_tf.get_forward_vector() * 8 + carla.Location(z=5)
        chase_rot = carla.Rotation(pitch=-15, yaw=av_tf.rotation.yaw)
        spectator.set_transform(carla.Transform(chase_loc, chase_rot))

        # Logging: distance and speeds
        dist = np.hypot(av_state['x'] - hdv_state['x'], av_state['y'] - hdv_state['y'])
        v_av = np.hypot(av_state['vx'], av_state['vy'])
        v_hdv = np.hypot(hdv_state['vx'], hdv_state['vy'])

        print(
            f"Iter {i:03d}: "
            f"Distance={dist:.2f} m, "
            f"v_av={v_av:.2f} m/s, v_hdv={v_hdv:.2f} m/s, "
            f"a_cmd={a_cmd:.2f} m/s², "
            f"throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}"
        )

        time.sleep(controller.dt)

    print("Simulation complete. Vehicles remain spawned in the environment.")


if __name__ == '__main__':
    main()
