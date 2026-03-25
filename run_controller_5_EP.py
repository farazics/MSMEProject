import time
import os
import csv
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


def run_one_episode(world, episode_idx, controller, n_steps=200, dt=0.1, log_dir="data"):
    """
    Run a single episode:
      - spawn AV/HDV
      - run n_steps of simulation
      - log per-step data to CSV
      - destroy actors at the end
    """
    # Make sure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"episode_{episode_idx:03d}.csv")

    # Get spawn points
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    # HDV spawns 25 m ahead along AV heading
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
        print(f"[Episode {episode_idx}] ERROR: Could not spawn one or both vehicles.")
        if av is not None:
            av.destroy()
        if hdv is not None:
            hdv.destroy()
        return

    # Set spectator view
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        av_tf.location + carla.Location(z=30),
        carla.Rotation(pitch=-90)
    ))

    print(f"[Episode {episode_idx}] Starting episode...")

    # Prepare CSV logger
    with open(log_path, "w", newline="") as csvfile:
        fieldnames = ["episode", "iter", "time",
                      "distance", "v_av", "v_hdv",
                      "a_cmd", "throttle", "brake"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, n_steps + 1):
            t = i * dt

            av_state = get_vehicle_state(av)
            hdv_state = get_vehicle_state(hdv)

            # HDV behavior: accelerate, then cruise, then brake to stop
            if i <= 50:
                hdv.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, steer=0.0))
            elif 51 <= i <= 100:
                hdv.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0, steer=0.0))
            else:
                hdv.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))

            # RL will replace nominal_acc with a learned value a_rl each timestep, but AdaptiveController stays unchanged and still enforces safety
            nominal_acc = 1.0
            a_cmd = controller.compute_control(av_state, hdv_state, nominal_acc)

            # Map acceleration to throttle/brake
            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)

            # Apply control to AV (pure longitudinal)
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

            # Compute metrics and log
            dist = np.hypot(av_state['x'] - hdv_state['x'],
                            av_state['y'] - hdv_state['y'])
            v_av = np.hypot(av_state['vx'], av_state['vy'])
            v_hdv = np.hypot(hdv_state['vx'], hdv_state['vy'])

            writer.writerow({
                "episode": episode_idx,
                "iter": i,
                "time": t,
                "distance": dist,
                "v_av": v_av,
                "v_hdv": v_hdv,
                "a_cmd": a_cmd,
                "throttle": throttle_cmd,
                "brake": brake_cmd
            })

            # Optional: also print a few steps for sanity
            if i % 20 == 0 or i == 1:
                print(
                    f"[Ep {episode_idx} | Step {i:03d}] "
                    f"dist={dist:.2f} m, v_av={v_av:.2f} m/s, v_hdv={v_hdv:.2f} m/s, "
                    f"a_cmd={a_cmd:.2f}, throttle={throttle_cmd:.2f}, brake={brake_cmd:.2f}"
                )

            time.sleep(dt)

    print(f"[Episode {episode_idx}] Episode finished. Destroying actors...")

    # Clean up actors
    av.destroy()
    hdv.destroy()

    # Short pause to let CARLA clean up
    time.sleep(1.0)


def main():
    N_EPISODES = 5
    N_STEPS = 200
    DT = 0.1

    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
    except RuntimeError as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        return
    except Exception as e:
        print(f"ERROR: Unexpected error connecting to CARLA: {e}")
        return

    # Optional: you can set synchronous mode here if you want more deterministic timing
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = DT
    # world.apply_settings(settings)

    # Initialize controller once and reuse (it keeps internal state)
    controller = AdaptiveController(
        time_headway=1.8,   # or 3.0 if you want, depending on your final choice
        standstill_gap=5.0,
        alpha=1.0,
        dt=DT
    )

    for ep in range(1, N_EPISODES + 1):
        # Option A: reuse world as-is
        # Option B: reload world for a fully fresh episode:
        # world = client.reload_world()
        run_one_episode(world, ep, controller, n_steps=N_STEPS, dt=DT, log_dir="data")

    print("All episodes completed.")


if __name__ == '__main__':
    main()
