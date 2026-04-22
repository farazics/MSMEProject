import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# Load long-run log
# -----------------------------------------
df = pd.read_csv("RL_long.csv")

# -----------------------------------------
# Global configuration
# -----------------------------------------
dt = 0.1            # must match your controller timestep
steps_per_ep = 100  # virtual episode length in steps

# Add a virtual episode index: 1..num_episodes
num_episodes = df["step"].max() // steps_per_ep
df["episode"] = ((df["step"] - 1) // steps_per_ep) + 1

# Ensure we have a run index column
if "run" not in df.columns and "run_index" not in df.columns:
    df["run_index"] = 1
elif "run" in df.columns:
    df = df.rename(columns={"run": "run_index"})


# -----------------------------------------
# Collision event extraction (1 per collision)
# -----------------------------------------
def add_collision_events(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    if "collision_count" in df_out.columns:
        # Per-step increment of the cumulative counter
        df_out["collision_delta"] = df_out["collision_count"].diff().fillna(0)

        # Mark the START of each collision (first positive delta after zero)
        df_out["collision_event"] = (
            (df_out["collision_delta"] > 0)
            & (df_out["collision_delta"].shift(fill_value=0) <= 0)
        ).astype(int)

        # Cumulative number of collision events
        df_out["collision_event_cum"] = df_out["collision_event"].cumsum()
    else:
        df_out["collision_event"] = 0
        df_out["collision_event_cum"] = 0
    return df_out


df = add_collision_events(df)


# -----------------------------------------
# Episode-level statistics (comfort + safety)
# -----------------------------------------
def episode_stats(ep_df: pd.DataFrame, ep_idx: int) -> None:
    a = ep_df["a_cmd_total"].values

    # Jerk: use column if present, else compute
    if "jerk" in ep_df.columns:
        jerk = ep_df["jerk"].values
    else:
        jerk = np.diff(a) / dt
        jerk = np.concatenate([[0.0], jerk])

    mean_speed = ep_df["av_speed"].mean()
    rms_accel = float(np.sqrt(np.mean(a ** 2)))
    rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
    max_decel = float(np.min(a))

    unsafe_steps = int((ep_df["time_headway"] < 1.0).sum()) if "time_headway" in ep_df.columns else 0

    # Collision events in this episode (1 per collision)
    total_collisions = 0
    if "collision_event" in ep_df.columns:
        total_collisions = int(ep_df["collision_event"].sum())

    print(f"\n=== Episode {ep_idx} ===")
    print(f"Steps: {len(ep_df)}")
    print(f"Mean speed : {mean_speed:.3f} m/s")
    print(f"RMS accel : {rms_accel:.3f} m/s^2")
    print(f"RMS jerk  : {rms_jerk:.3f} m/s^3")
    print(f"Max decel : {max_decel:.3f} m/s^2")
    print(f"Unsafe headway (<1s): {unsafe_steps} steps")
    print(f"Collision events    : {total_collisions}")


for ep in range(1, num_episodes + 1):
    ep_df = df[df["episode"] == ep]
    if len(ep_df) == 0:
        continue
    episode_stats(ep_df, ep)


# -----------------------------------------
# Idea B: cumulative collision events vs time across runs
# -----------------------------------------
plt.figure()

for run_id, run_df in df.groupby("run_index"):
    plt.step(run_df["step"], run_df["collision_event_cum"],
             where="post", label=f"Run {run_id}", alpha=0.8)

plt.xlabel("Step")
plt.ylabel("Number of collision events")
plt.title("RL collisions vs. time across runs\n(1 unit = 1 collision event)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()
