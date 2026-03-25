import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)  # Adjust timeout if needed

    world = client.get_world()  # Uses the map currently loaded in simulator

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is not None:
            print("Vehicle spawned successfully!")
            # Move camera above vehicle
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=30),
                carla.Rotation(pitch=-90)))
            break

    if vehicle is None:
        print("Failed to spawn vehicle at any spawn point.")
        return

    time.sleep(10)  # Keep script alive to see the vehicle
    vehicle.destroy()
    print("Vehicle destroyed and script finished.")

if __name__ == '__main__':
    main()
