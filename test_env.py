# test_env.py
"""
Simple test: run the Gym environment with RANDOM actions.
Goal: verify environment doesn't crash and outputs sensible values.

This script also launches the CARLA server if needed.
"""

import numpy as np
import subprocess
import time
import sys
import os
from carla_acc_gym_env import CarlaACCEnv

def launch_carla(carla_path=None, port=2000):
    """
    Launch CARLA server if not already running.
    
    Args:
        carla_path: path to CARLA executable (e.g., ./CarlaUE4.exe on Windows)
        port: CARLA server port (default 2000)
    """
    print("Checking if CARLA is already running...")
    
    # Try to connect to see if CARLA is running
    try:
        import carla
        client = carla.Client('localhost', port)
        client.set_timeout(2.0)
        client.get_world()
        print("✓ CARLA is already running on port 2000")
        return None
    except:
        print("✗ CARLA not running. Launching CARLA server...")
    
    # Find CARLA executable
    if carla_path is None:
        # Try common Windows paths
        possible_paths = [
            "CarlaUE4.exe",
            "./CarlaUE4.exe",
            "C:\\CARLA\\WindowsNoEditor\\CarlaUE4.exe",
        ]
        
        # Also check for Linux/Mac
        if sys.platform != "win32":
            possible_paths.extend([
                "./CarlaUE4.sh",
                "/opt/carla/CarlaUE4.sh",
            ])
        
        for path in possible_paths:
            if os.path.exists(path):
                carla_path = path
                break
    
    if carla_path is None:
        print("\n⚠️  Could not find CARLA executable.")
        print("Please ensure CARLA is running manually before continuing.")
        print("Or specify the path manually:")
        print("  python test_env.py --carla-path /path/to/CarlaUE4.exe")
        input("Press Enter once CARLA is running...")
        return None
    
    print(f"Launching CARLA from: {carla_path}")
    
    try:
        if sys.platform == "win32":
            # Windows: launch in background
            subprocess.Popen([carla_path, '-quality-level=Low'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        else:
            # Linux/Mac: launch in background
            subprocess.Popen([carla_path, '-quality-level=Low'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        
        print("CARLA launched. Waiting for server to initialize (30 seconds)...")
        
        # Wait for CARLA to start
        for i in range(30):
            try:
                import carla
                client = carla.Client('localhost', port)
                client.set_timeout(2.0)
                client.get_world()
                print(f"✓ CARLA ready after {i+1} seconds!")
                return client
            except:
                print(f"  Waiting... ({i+1}/30)", end='\r')
                time.sleep(1)
        
        print("\n⚠️  CARLA did not respond after 30 seconds. Trying anyway...")
        
    except Exception as e:
        print(f"Error launching CARLA: {e}")
        print("Please launch CARLA manually and try again.")
    
    return None


def main():
    print("=" * 60)
    print("CARLA ACC Environment Test")
    print("=" * 60)
    
    # Launch CARLA if needed
    carla_process = launch_carla()
    
    print("\nCreating environment...")
    env = CarlaACCEnv(dt=0.1, max_steps=50)

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    
    # DEBUG: Check if actors exist
    print(f"\nDEBUG: env.ego = {env.ego}")
    print(f"DEBUG: env.hdv = {env.hdv}")

    print("\nRunning 50 steps with random actions...")
    done = False
    step = 0

    while not done:
        # Choose a random action from the action space
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, done, info = env.step(action)

        step += 1
        
        # Print every 10 steps
        if step % 10 == 0 or step == 1:
            print(
                f"Step {step:3d} | "
                f"gap={info['gap']:.2f}m | "
                f"v_av={info['v_av']:.2f}m/s | "
                f"v_hdv={info['v_hdv']:.2f}m/s | "
                f"headway={info['headway']:.2f}s | "
                f"reward={reward:.3f} | "
                f"a_rl={info['a_rl']:.2f}"
            )

    print("\n✓ Test completed successfully!")
    env.close()

if __name__ == "__main__":
    main()
