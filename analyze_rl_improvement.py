import pandas as pd  # type: ignore
import numpy as np

# Load long-run log
df = pd.read_csv("RL_long.csv")

dt = 0.1
steps_per_ep = 100
num_episodes = df["step"].max() // steps_per_ep

# Add a virtual episode index: 1..num_episodes
df["episode"] = ((df["step"] - 1) // steps_per_ep) + 1

def episode_stats(ep_df: pd.DataFrame, ep_idx: int):
    a = ep_df["a_cmd_total"].values
    if "jerk" in ep_df.columns:
        jerk = ep_df["jerk"].values
    else:
        jerk = np.diff(a) / dt
        jerk = np.concatenate([[0.0], jerk])

    mean_speed = ep_df["av_speed"].mean()
    rms_accel = float(np.sqrt(np.mean(a ** 2)))
    rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
    max_decel = float(np.min(a))

    unsafe_steps = int((ep_df["time_headway"] < 1.0).sum())

    total_collisions = 0
    if "collision_count" in ep_df.columns:
        # collision_count is cumulative; take last minus first for this block
        total_collisions = int(ep_df["collision_count"].iloc[-1] -
                               ep_df["collision_count"].iloc[0])

    print(f"\n=== Episode {ep_idx} ===")
    print(f"Steps: {len(ep_df)}")
    print(f"Mean speed         : {mean_speed:.3f} m/s")
    print(f"RMS accel          : {rms_accel:.3f} m/s^2")
    print(f"RMS jerk           : {rms_jerk:.3f} m/s^3")
    print(f"Max decel          : {max_decel:.3f} m/s^2")
    print(f"Unsafe headway (<1s): {unsafe_steps} steps")
    print(f"Collisions          : {total_collisions}")

for ep in range(1, num_episodes + 1):
    ep_df = df[df["episode"] == ep]
    if len(ep_df) == 0:
        continue
    episode_stats(ep_df, ep)
