import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
dt = 0.1
steps_per_ep = 100  # adjust if your episode length changes

# --- Load long-run log ---
df = pd.read_csv("RL_long.csv")

# Add a virtual episode index: 1..num_episodes
num_episodes = df["step"].max() // steps_per_ep
df["episode"] = ((df["step"] - 1) // steps_per_ep) + 1

# Containers for episode-level metrics (for plotting)
episode_ids = []
rms_accel_list = []
rms_jerk_list = []
unsafe_steps_list = []
collision_events_list = []
mean_speed_list = []
max_decel_list = []

def episode_stats(ep_df: pd.DataFrame, ep_idx: int):
    # Acceleration
    a = ep_df["a_cmd_total"].values

    # Jerk (compute if not present)
    if "jerk" in ep_df.columns:
        jerk = ep_df["jerk"].values
    else:
        jerk = np.diff(a) / dt
        jerk = np.concatenate([[0.0], jerk])

    mean_speed = ep_df["av_speed"].mean()
    rms_accel = float(np.sqrt(np.mean(a ** 2)))
    rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
    max_decel = float(np.min(a))

    unsafe_steps = 0
    if "time_headway" in ep_df.columns:
        unsafe_steps = int((ep_df["time_headway"] < 1.0).sum())

    total_collisions = 0
    if "collision_count" in ep_df.columns:
        # Convert cumulative counter to per-event increments (1 per collision)
        events = ep_df["collision_count"].diff().fillna(0)
        events = events.clip(lower=0).astype(int)
        total_collisions = int(events.sum())

    # Print stats for this episode
    print(f"\n=== Episode {ep_idx} ===")
    print(f"Steps: {len(ep_df)}")
    print(f"Mean speed : {mean_speed:.3f} m/s")
    print(f"RMS accel : {rms_accel:.3f} m/s^2")
    print(f"RMS jerk : {rms_jerk:.3f} m/s^3")
    print(f"Max decel : {max_decel:.3f} m/s^2")
    print(f"Unsafe headway (<1s): {unsafe_steps} steps")
    print(f"Collision events : {total_collisions}")

    # Store metrics for plotting
    episode_ids.append(ep_idx)
    rms_accel_list.append(rms_accel)
    rms_jerk_list.append(rms_jerk)
    unsafe_steps_list.append(unsafe_steps)
    collision_events_list.append(total_collisions)
    mean_speed_list.append(mean_speed)
    max_decel_list.append(max_decel)

# --- Loop over episodes and collect stats ---
for ep in range(1, num_episodes + 1):
    ep_df = df[df["episode"] == ep]
    if len(ep_df) == 0:
        continue
    episode_stats(ep_df, ep)

# --- Plots over episodes ---

# 1. RMS acceleration per episode
plt.figure()
plt.plot(episode_ids, rms_accel_list, marker="o")
plt.xlabel("Episode")
plt.ylabel("RMS acceleration [m/s^2]")
plt.title("RMS acceleration vs. episode")
plt.grid(True)

# 2. RMS jerk per episode
plt.figure()
plt.plot(episode_ids, rms_jerk_list, marker="o")
plt.xlabel("Episode")
plt.ylabel("RMS jerk [m/s^3]")
plt.title("RMS jerk vs. episode")
plt.grid(True)

# 3. Unsafe headway steps per episode
plt.figure()
plt.plot(episode_ids, unsafe_steps_list, marker="o")
plt.xlabel("Episode")
plt.ylabel("Steps with time_headway < 1 s")
plt.title("Unsafe headway steps vs. episode")
plt.grid(True)

# 4. Collision events per episode
plt.figure()
plt.bar(episode_ids, collision_events_list)
plt.xlabel("Episode")
plt.ylabel("Number of collision events")
plt.title("Collision events vs. episode")
plt.grid(True, axis="y")

# Optional: mean speed per episode
plt.figure()
plt.plot(episode_ids, mean_speed_list, marker="o")
plt.xlabel("Episode")
plt.ylabel("Mean speed [m/s]")
plt.title("Mean speed vs. episode")
plt.grid(True)

plt.show()
