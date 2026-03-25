import time
import csv
import math
import random
import numpy as np
import carla

from adaptive_controller import AdaptiveController
from Stanley_controller import StanleyController


def get_vehicle_state(vehicle):
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return {
        "x": tf.location.x,
        "y": tf.location.y,
        "vx": vel.x,
        "vy": vel.y,
        "yaw": math.radians(tf.rotation.yaw),
    }


def main():
    # ---------- Connect to CARLA ----------
    try:
        print("Connecting to CARLA server...")
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)
        world = client.get_world()
        print(f"Connected! World: {world.get_map().name}")
    except Exception as e:
        print(f"ERROR: Could not connect to CARLA server: {e}")
        return

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    # Asynchronous mode (you may switch to sync later)
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    # ---------- Spawn ego vehicle ----------
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find("vehicle.tesla.model3")

    av = world.try_spawn_actor(av_bp, av_tf)
    if av is None:
        print("ERROR: Could not spawn AV.")
        return

    # Ego is NOT on autopilot
    # av.set_autopilot(True, tm_port)

    # ---------- Spawn NPC vehicles ----------
    vehicle_bps = bp_lib.filter("*vehicle*")
    spawn_points = world.get_map().get_spawn_points()
    npc_vehicles = []

    num_npcs = 00
    random.shuffle(spawn_points)

    for sp, bp in zip(spawn_points,
                      random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None and npc.id != av.id:
            npc.set_autopilot(True, tm_port)
            npc_vehicles.append(npc)

    print(f"Spawned {len(npc_vehicles)} NPC vehicles registered to TM.")

    # ---------- Controllers ----------
    long_controller = AdaptiveController(
        time_headway=3.0,
        standstill_gap=5.0,
        alpha=1.0,
        dt=0.1,
    )
    lat_controller = StanleyController()

    # ---------- Spectator ----------
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            av_tf.location + carla.Location(z=30.0),
            carla.Rotation(pitch=-90.0),
        )
    )

    # ---------- Logging ----------
    log_filename = "av_cbfpcca_stanley_lane_cte.csv"
    with open(log_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "step",
                "av_x",
                "av_y",
                "av_yaw",
                "av_speed",
                "a_cmd",
                "throttle_cmd",
                "brake_cmd",
                "steer_cmd",
                "cte",
                "lead_id",
                "lead_x",
                "lead_y",
                "lead_speed",
                "distance_headway",
                "time_headway",
            ]
        )

        print("Starting CARLA sim with CBF longitudinal + lane‑based Stanley...")

        v_des = 12.0     # global desired speed [m/s]
        nominal_acc = 1.0
        max_steps = 2000

        for i in range(1, max_steps + 1):
            # ---------- Ego state ----------
            av_state = get_vehicle_state(av)
            v = math.hypot(av_state["vx"], av_state["vy"])

            # ---------- Lane‑based reference waypoint ----------
            ego_tf = av.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()

            ego_wp = world.get_map().get_waypoint(
                ego_loc, project_to_road=True
            )

            lookahead_dist = 8.0  # m; tune 5–15
            next_wps = ego_wp.next(lookahead_dist)
            if next_wps:
                target_wp = next_wps[0]
            else:
                target_wp = ego_wp

            target_loc = target_wp.transform.location
            path_yaw = math.radians(target_wp.transform.rotation.yaw)

            # ---------- Find closest leader in same lane ----------
            vehicles = world.get_actors().filter("vehicle.*")
            leader = None
            leader_dist = float("inf")

            for veh in vehicles:
                if veh.id == av.id:
                    continue

                tf_v = veh.get_transform()
                loc_v = tf_v.location

                rel = loc_v - ego_loc
                dot = (
                    rel.x * ego_forward.x
                    + rel.y * ego_forward.y
                    + rel.z * ego_forward.z
                )
                if dot <= 0:
                    continue

                veh_wp = world.get_map().get_waypoint(
                    loc_v, project_to_road=True
                )
                if (
                    veh_wp.road_id != ego_wp.road_id
                    or veh_wp.lane_id != ego_wp.lane_id
                ):
                    continue

                dist = math.sqrt(rel.x ** 2 + rel.y ** 2 + rel.z ** 2)
                if 1.0 < dist < leader_dist:
                    leader_dist = dist
                    leader = veh

            hdv_state = None
            if leader is not None:
                tf_l = leader.get_transform()
                vel_l = leader.get_velocity()
                hdv_state = {
                    "x": tf_l.location.x,
                    "y": tf_l.location.y,
                    "vx": vel_l.x,
                    "vy": vel_l.y,
                    "yaw": math.radians(tf_l.rotation.yaw),
                }

            # ---------- Speed cap while tuning ----------
            v_des_local = min(v_des, 10.0)  # 10 m/s cap

            # ---------- Longitudinal control ----------
            if hdv_state is not None:
                a_cmd = long_controller.compute_control(
                    av_state, hdv_state, nominal_acc
                )
            else:
                k_p = 0.5
                a_cmd = k_p * (v_des_local - v)
                a_cmd = float(
                    np.clip(
                        a_cmd,
                        long_controller.min_acceleration,
                        long_controller.max_acceleration,
                    )
                )

            throttle_cmd = max(a_cmd, 0.0)
            brake_cmd = max(-a_cmd, 0.0)
            throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
            brake_cmd = float(np.clip(brake_cmd, 0.0, 1.0))

            # ---------- Lateral control: lane‑based Stanley ----------
            steer_cmd = lat_controller.compute_control(
                av, v, target_loc, path_yaw
            )
            cte = lat_controller.last_cte

            # ---------- Apply control ----------
            av.apply_control(
                carla.VehicleControl(
                    throttle=throttle_cmd,
                    brake=brake_cmd,
                    steer=steer_cmd,
                    manual_gear_shift=False,
                )
            )

            # ---------- Update spectator ----------
            av_tf_curr = av.get_transform()
            chase_loc = (
                av_tf_curr.location
                - av_tf_curr.get_forward_vector() * 8.0
                + carla.Location(z=5.0)
            )
            chase_rot = carla.Rotation(
                pitch=-15.0, yaw=av_tf_curr.rotation.yaw
            )
            spectator.set_transform(carla.Transform(chase_loc, chase_rot))

            # ---------- Logging ----------
            lead_id = -1
            lead_x = lead_y = lead_speed = float("nan")
            dist_headway = time_headway = float("nan")

            if hdv_state is not None:
                lead_id = leader.id
                lead_x = hdv_state["x"]
                lead_y = hdv_state["y"]
                lead_speed = math.hypot(
                    hdv_state["vx"], hdv_state["vy"]
                )
                dist_headway = leader_dist
                if v > 0.1:
                    time_headway = dist_headway / v

            writer.writerow(
                [
                    i,
                    av_state["x"],
                    av_state["y"],
                    av_state["yaw"],
                    v,
                    a_cmd,
                    throttle_cmd,
                    brake_cmd,
                    steer_cmd,
                    cte,
                    lead_id,
                    lead_x,
                    lead_y,
                    lead_speed,
                    dist_headway,
                    time_headway,
                ]
            )

            print(
                f"Iter {i:04d}: "
                f"v={v:.2f} m/s, a_cmd={a_cmd:.2f}, "
                f"thr={throttle_cmd:.2f}, brk={brake_cmd:.2f}, "
                f"steer={steer_cmd:.2f}, cte={cte:.2f}, "
                f"lead_id={lead_id}, dist={dist_headway:.2f} m, "
                f"t_headway={time_headway:.2f} s"
            )

            time.sleep(long_controller.dt)

    print(f"Simulation complete. Log saved to {log_filename}")


if __name__ == "__main__":
    main()
