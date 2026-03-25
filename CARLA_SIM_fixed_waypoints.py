import time
import csv
import math
import random

import numpy as np
import carla

import os
import sys

# Path to CARLA root (folder that contains PythonAPI)
CARLA_ROOT = r"C:\Users\FarazShaikh(WovenbyT\Downloads\CARLA_0.9.14\WindowsNoEditor"

# 1) Add PythonAPI to sys.path so we can import "carla" and its agents
python_api_root = os.path.join(CARLA_ROOT, "PythonAPI")
if python_api_root not in sys.path:
    sys.path.append(python_api_root)

# 2) Add the folder that contains the "carla" package (so 'carla.agents' works)
carla_python_package = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if carla_python_package not in sys.path:
    sys.path.append(carla_python_package)

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from adaptive_controller import AdaptiveController
from Stanley_controller import StanleyController


# ---------- Helpers ----------

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


def get_front_axle(av, wheel_base=2.8):
    tf = av.get_transform()
    x = tf.location.x
    y = tf.location.y
    yaw = math.radians(tf.rotation.yaw)
    fx = x + wheel_base * math.cos(yaw)
    fy = y + wheel_base * math.sin(yaw)
    return fx, fy


def find_closest_route_index(route_wps, fx, fy, last_idx=0, window=80):
    start = max(last_idx - window, 0)
    end = min(last_idx + window, len(route_wps) - 1)
    best_idx = start
    best_dist2 = float("inf")
    for i in range(start, end + 1):
        loc = route_wps[i].transform.location
        dx = fx - loc.x
        dy = fy - loc.y
        d2 = dx * dx + dy * dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_idx = i
    return best_idx


def draw_route(world, route_wps):
    debug = world.debug
    for i in range(len(route_wps) - 1):
        loc1 = route_wps[i].transform.location
        loc2 = route_wps[i + 1].transform.location
        debug.draw_line(
            loc1 + carla.Location(z=0.1),
            loc2 + carla.Location(z=0.1),
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=0.0,
        )
        if i % 5 == 0:
            debug.draw_point(
                loc1 + carla.Location(z=0.2),
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=0.0,
            )


# ---------- Main ----------

def main():
    # ---------- Connect to CARLA ----------
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Connected! World: {carla_map.name}")

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    settings = world.get_settings()
    settings.synchronous_mode = False  # keep async for now
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()

    # ---------- Spawn ego vehicle ----------
    spawns = carla_map.get_spawn_points()
    if not spawns:
        print("ERROR: no spawn points found.")
        return

    # Use fixed spawn for reproducibility
    start_tf = spawns[0]
    goal_tf = spawns[10] if len(spawns) > 10 else spawns[-1]

    av_bp = bp_lib.find("vehicle.tesla.model3")
    av = world.try_spawn_actor(av_bp, start_tf)
    if av is None:
        print("ERROR: could not spawn AV.")
        return

    # ---------- Optional NPCs ----------
    vehicle_bps = bp_lib.filter("*vehicle*")
    spawn_points = carla_map.get_spawn_points()
    npc_vehicles = []
    num_npcs = 0
    random.shuffle(spawn_points)
    for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None and npc.id != av.id:
            npc.set_autopilot(True, tm_port)
            npc_vehicles.append(npc)
    print(f"Spawned {len(npc_vehicles)} NPC vehicles.")

    # ---------- Build fixed global route ----------
    # Use CARLA's GlobalRoutePlanner
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

    sampling_resolution = 1.0  # meters
    dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    start_wp = carla_map.get_waypoint(
        start_tf.location, project_to_road=True
    )
    goal_wp = carla_map.get_waypoint(
        goal_tf.location, project_to_road=True
    )

    route = grp.trace_route(
        start_wp.transform.location,
        goal_wp.transform.location,
    )

    route_wps = [wp for (wp, road_option) in route]
    print(f"Global route length: {len(route_wps)} waypoints")

    draw_route(world, route_wps)

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
            start_tf.location + carla.Location(z=30.0),
            carla.Rotation(pitch=-90.0),
        )
    )

    # ---------- Logging ----------
    log_filename = "av_cbfpcca_stanley_fixed_route_cte.csv"
    csvfile = open(log_filename, mode="w", newline="")
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
            "closest_idx",
            "lead_id",
            "lead_x",
            "lead_y",
            "lead_speed",
            "distance_headway",
            "time_headway",
        ]
    )

    print("Starting CARLA sim with fixed-route lane-based Stanley...")

    v_des = 12.0  # m/s
    nominal_acc = 1.0
    max_steps = 2000
    dt = long_controller.dt

    # For nearest search
    closest_idx = 0
    N = 12  # look-ahead index (~12 m)

    try:
        for i in range(1, max_steps + 1):
            # ---------- Ego state ----------
            av_state = get_vehicle_state(av)
            v = math.hypot(av_state["vx"], av_state["vy"])

            # ---------- Lateral reference from fixed route ----------
            fx, fy = get_front_axle(av, wheel_base=lat_controller.wheel_base)

            closest_idx = find_closest_route_index(
                route_wps, fx, fy, last_idx=closest_idx, window=80
            )

            # Segment for heading
            seg_i = closest_idx
            if seg_i >= len(route_wps) - 1:
                seg_i = len(route_wps) - 2
            wp_i = route_wps[seg_i]
            wp_ip1 = route_wps[seg_i + 1]

            p0 = wp_i.transform.location
            p1 = wp_ip1.transform.location
            path_yaw = math.atan2(p1.y - p0.y, p1.x - p0.x)

            # Look-ahead target
            target_idx = min(closest_idx + N, len(route_wps) - 1)
            target_wp = route_wps[target_idx]
            target_loc = target_wp.transform.location

            # ---------- Leader search (same as your original) ----------
            ego_tf = av.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()
            ego_wp = carla_map.get_waypoint(
                ego_loc, project_to_road=True
            )

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
                veh_wp = carla_map.get_waypoint(
                    loc_v, project_to_road=True
                )
                if (
                    veh_wp.road_id != ego_wp.road_id
                    or veh_wp.lane_id != ego_wp.lane_id
                ):
                    continue
                dist = math.sqrt(
                    rel.x ** 2 + rel.y ** 2 + rel.z ** 2
                )
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
            v_des_local = min(v_des, 10.0)

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

            # ---------- Lateral control: fixed-route Stanley ----------
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

            # ---------- Spectator chase cam ----------
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
                    closest_idx,
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
                f"idx={closest_idx}, lead_id={lead_id}, "
                f"dist={dist_headway:.2f}, t_headway={time_headway:.2f}"
            )

            time.sleep(dt)

        print(f"Simulation complete. Log saved to {log_filename}")

    finally:
        csvfile.close()
        # Clean up actors
        print("Destroying actors...")
        for veh in npc_vehicles:
            veh.destroy()
        av.destroy()
        print("Done.")


if __name__ == "__main__":
    main()
