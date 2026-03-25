# run_controller_with_debug_and_chasecam.py

import carla
import time
import numpy as np
from adaptive_controller import AdaptiveController

def get_vehicle_state(v):
    """Retrieve a vehicle's state: position, velocity, and heading."""
    tf = v.get_transform()
    vel = v.get_velocity()
    return {
        'x': tf.location.x,
        'y': tf.location.y,
        'vx': vel.x,
        'vy': vel.y,
        'yaw': np.deg2rad(tf.rotation.yaw)
    }

def visualize_spawn_points(world, duration=3.0):
    """Draw all spawn points and their forward direction for user inspection."""
    spawns = world.get_map().get_spawn_points()
    print(f"Total spawn points: {len(spawns)}")
    for idx, transform in enumerate(spawns):
        world.debug.draw_string(
            transform.location,
            text=str(idx),
            life_time=duration,
            color=carla.Color(r=255, g=0, b=0)
        )
        world.debug.draw_arrow(
            transform.location,
            transform.location + transform.get_forward_vector() * 5.0,
            life_time=duration,
            thickness=0.1,
            arrow_size=0.2,
            color=carla.Color(r=0, g=255, b=0)
        )
    print(f"Spawn points visualized for {duration} seconds. Press Tab to enter free camera and note visible indices.")
    time.sleep(duration)

def main():
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    for _ in range(20):
        try:
            world = client.get_world()
            break
        except RuntimeError:
            time.sleep(1.0)
    else:
        print("ERROR: Could not connect to CARLA server.")
        return

    # Load map
    world = client.load_world('Town05')
    time.sleep(2.0)
    print("Loaded map:", world.get_map().name)

    # Visualize spawn points and prompt for AV index
    visualize_spawn_points(world, duration=30.0)
    total = len(world.get_map().get_spawn_points())
    av_idx = int(input(f"Enter AV spawn index (0–{total-1}): "))
    av_tf = world.get_map().get_spawn_points()[av_idx]
    print(f"Selected AV spawn index: {av_idx}")

    # Compute HDV spawn transform offset 60 m ahead of AV
    offset = 35.0
    forward = av_tf.get_forward_vector()
    hdv_location = av_tf.location + forward * offset
    hdv_tf = carla.Transform(hdv_location, av_tf.rotation)
    print(f"HDV will spawn {offset} m ahead at: {hdv_location}")

    # Spawn AV and HDV
    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')
    hdv_bp = bp_lib.filter('vehicle.audi.tt')[0]

    av = world.try_spawn_actor(av_bp, av_tf)
    hdv = world.try_spawn_actor(hdv_bp, hdv_tf)
    if av is None or hdv is None:
        print("ERROR: Failed to spawn AV or HDV.")
        return
    print("Actors spawned successfully.")

    # Center spectator above AV for initial view
    spectator = world.get_spectator()
    av_loc = av.get_transform().location
    spectator.set_transform(carla.Transform(
        av_loc + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    # Start HDV moving forward at constant throttle
    hdv.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

    # Initialize adaptive controller
    controller = AdaptiveController(dsafe=20.0, h_threshold=0.1, dt=0.1)

    print("Starting control loop for 100 iterations.")
    for i in range(1, 151):
        av_state = get_vehicle_state(av)
        hdv_state = get_vehicle_state(hdv)

        # Log distance and commands
        distance = np.hypot(
            av_state['x'] - hdv_state['x'],
            av_state['y'] - hdv_state['y']
        )

        v_av = np.hypot(av_state['vx'], av_state['vy'])
        v_hdv = np.hypot(hdv_state['vx'], hdv_state['vy'])
        rel_speed = v_av - v_hdv   # >0 means AV is closing in

        # Compute adaptive commands: nominal (1.0, 0.0)
        a_cmd, steer_cmd = controller.compute_control(
            av_state, hdv_state, (2.0, 1.0)
        )

        # ---------- HARD SAFETY LAYER ----------
        # If we are inside the safe distance, force strong braking
        
        if distance < controller.dsafe:
            a_cmd = -3.0  # tune this; more negative = stronger brake

        # If we are close to the safe distance and still closing in, block acceleration
        elif distance < controller.dsafe + 5.0 and rel_speed > 0.0:
            a_cmd = min(a_cmd, 0.0)  # don't allow positive acceleration

        # Apply to AV
        #throttle = max(a_cmd, 0.0)
        throttle = max(min(a_cmd, 1.0), 0.0)  # [0,1]

        #brake = max(-a_cmd, 0.0)
        brake = max(min(-a_cmd, 1.0), 0.0)    # [0,1]
        
        av.apply_control(carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer_cmd
        ))

        # Chase-camera: position spectator behind and above AV
        av_tf = av.get_transform()
        chase_loc = av_tf.location - av_tf.get_forward_vector() * 10 + carla.Location(z=5)
        chase_rot = carla.Rotation(pitch=-15.0, yaw=av_tf.rotation.yaw)
        spectator.set_transform(carla.Transform(chase_loc, chase_rot))

        # # Log distance and commands
        # distance = np.hypot(
        #     av_state['x'] - hdv_state['x'],
        #     av_state['y'] - hdv_state['y']
        # )
        # print(f"[{i:03d}] dist={distance:.2f}, a_cmd={a_cmd:.2f}, steer={steer_cmd:.2f}")
        print(f"[{i:03d}] dist={distance:.2f}, v_av={v_av:.2f}, v_hdv={v_hdv:.2f}, a_cmd={a_cmd:.2f}, steer={steer_cmd:.2f}")

        time.sleep(controller.dt)







    # # Apply to AV (same as before, maybe clamp)
    # throttle = max(min(a_cmd, 1.0), 0.0)  # [0,1]
    # brake = max(min(-a_cmd, 1.0), 0.0)    # [0,1]
    # av.apply_control(carla.VehicleControl(
    #     throttle=throttle,
    #     brake=brake,
    #     steer=steer_cmd
    # ))

    # # Chase-camera (unchanged)
    # av_tf = av.get_transform()
    # chase_loc = av_tf.location - av_tf.get_forward_vector() * 10 + carla.Location(z=5)
    # chase_rot = carla.Rotation(pitch=-15.0, yaw=av_tf.rotation.yaw)
    # spectator.set_transform(carla.Transform(chase_loc, chase_rot))

    # print(f"[{i:03d}] dist={distance:.2f}, v_av={v_av:.2f}, v_hdv={v_hdv:.2f}, a_cmd={a_cmd:.2f}, steer={steer_cmd:.2f}")

    # time.sleep(controller.dt)



    # Cleanup
    print("Completed 500 iterations. Cleaning up actors.")
    av.destroy()
    hdv.destroy()
    print("Done.")

if __name__ == '__main__':
    main()
