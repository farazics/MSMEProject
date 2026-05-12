import time
import csv
import math
import random
import os

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


def get_next_run_filename(prefix="CBF_controller_run", folder="."):
    """Auto-increment run number: CBF_controller_run1.csv, run2.csv, ..."""
    i = 1
    while os.path.exists(os.path.join(folder, f"{prefix}{i}.csv")):
        i += 1
    return os.path.join(folder, f"{prefix}{i}.csv")


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

    actors_to_destroy = []

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print("ERROR: No spawn points found in map.")
        return

    av_tf = spawns[0]

    bp_lib = world.get_blueprint_library()
    av_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle_bps = bp_lib.filter('*vehicle*')

    av = world.try_spawn_actor(av_bp, av_tf)
    if av is None:
        print("ERROR: Could not spawn AV.")
        return
    av.set_autopilot(True, tm_port)
    actors_to_destroy.append(av)

    spawn_points = world.get_map().get_spawn_points()
    npc_vehicles = []
    num_npcs = 35
    random.shuffle(spawn_points)
    for sp, bp in zip(spawn_points, random.choices(vehicle_bps, k=num_npcs)):
        npc = world.try_spawn_actor(bp, sp)
