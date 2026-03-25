# debug_spawn_points.py

import carla, time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    spawn_points = world.get_map().get_spawn_points()
    print(f"Total spawn points: {len(spawn_points)}")

    for i, transform in enumerate(spawn_points):
        world.debug.draw_string(
            transform.location,
            text=str(i),
            life_time=60.0,
            color=carla.Color(r=255, g=0, b=0)
        )
        world.debug.draw_arrow(
            transform.location,
            transform.location + transform.get_forward_vector() * 5.0,
            life_time=60.0,
            thickness=0.1,
            arrow_size=0.2,
            color=carla.Color(r=0, g=255, b=0)
        )
    print("Spawn points visualized for 60s. Use free camera (Tab) to fly around and note indices on roads.")
    time.sleep(60)

if __name__ == "__main__":
    main()
