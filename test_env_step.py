# from carla_acc_env import CarlaACCEnv


# def main():
#     env = CarlaACCEnv(dt=0.1, max_steps=200)  # shorter episode for test
#     obs, info = env.reset()
#     print("Initial obs:", obs)

#     for t in range(10):
#         # random action in [-3, 3]
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         print(f"Step {t+1}: obs={obs}, reward={reward}, done={done}")
#         if done:
#             break

#     env.close()


# if __name__ == "__main__":
#     main()


import time
from carla_acc_env import CarlaACCEnv

def main():
    env = CarlaACCEnv(dt=0.1, max_steps=200)  # full 20 seconds
    obs, info = env.reset()
    print("Initial obs:", obs)

    for t in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {t+1}: obs={obs}, reward={reward}, done={done}")

        # Slow it down so you can see motion in the CARLA window
        time.sleep(0.1)  # 0.1–0.2 seconds is usually comfortable

        if done:
            break

    # Keep the window open until you press Enter
    input("Press Enter to close env and exit...")
    env.close()

if __name__ == "__main__":
    main()
