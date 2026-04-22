import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# ========= Load long-run log =========

df = pd.read_csv("RL_long.csv")

# Time step used in simulation
dt = 0.1

# If you do NOT have an "episode" column already, you can create one here.
# Otherwise, comment this out.
# steps_per_ep = 100
# df["episode"] = ((df["step"] - 1) // steps_per_ep) + 1

# ========= Helpers =========

def smooth(series: pd.Series, window: int = 50) -> pd.Series:
    """Centered rolling mean for prettier, less noisy plots."""
    return series.rolling(window=window, center=True).mean()

def add_collision_events(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cumulative collision_count into:
      - collision_event: number of new events at each step (usually 0 or 1)
      - collision_cum: cumulative number of events per run
      - collision_cum_clamped: cumulative number of events, clipped to 1
    """
    if "collision_count" not in run_df.columns:
        run_df["collision_event"] = 0
        run_df["collision_cum"] = 0
        run_df["collision_cum_clamped"] = 0
        return run_df

    # difference of cumulative counter gives events per step
    events = run_df["collision_count"].diff().fillna(0).clip(lower=0).astype(int)
    run_df["collision_event"] = events
    run_df["collision_cum"] = events.cumsum()
    run_df["collision_cum_clamped"] = run_df["collision_cum"].clip(upper=1)
    return run_df

# apply collision processing per run
if "run" in df.columns:
    df = df.groupby("run", group_keys=False).apply(add_collision_events)
else:
    df = add_collision_events(df)

# ========= Per-episode numeric stats (printed) =========

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

    # 0/1 collision events per step, sum over episode
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
    print(f"Collision events : {total_collisions}")

if "episode" in df.columns:
    num_episodes = int(df["episode"].max())
    for ep in range(1, num_episodes + 1):
        ep_df = df[df["episode"] == ep]
        if len(ep_df) == 0:
            continue
        episode_stats(ep_df, ep)

# ========= Per-run aggregate stats (for clean plots) =========

run_stats = []

if "run" in df.columns:
    for run_id, run_df in df.groupby("run"):
        mean_speed = run_df["av_speed"].mean()
        rms_accel = float(np.sqrt(np.mean(run_df["a_cmd_total"] ** 2)))
        mean_headway = run_df["time_headway"].mean()
        total_collisions = int(run_df["collision_event"].sum())
        run_stats.append((run_id, mean_speed, rms_accel,
                          mean_headway, total_collisions))

    run_stats = pd.DataFrame(
        run_stats,
        columns=["run", "mean_speed", "rms_accel", "mean_headway", "collision_events"]
    )

    print("\n=== Per-run summary ===")
    print(run_stats)

# ========= Plots =========

if "run" in df.columns:
    # 1) Smoothed speed vs time for each run
    plt.figure()
    for run_id, run_df in df.groupby("run"):
        sp = smooth(run_df["av_speed"], window=50)
        plt.plot(run_df["step"], sp, label=f"Run {run_id}", alpha=0.9)
    plt.xlabel("Step")
    plt.ylabel("Speed [m/s]")
    plt.title("Smoothed speed vs. time across runs")
    plt.legend()
    plt.grid(True)

    # 2) Smoothed acceleration vs time for each run
    plt.figure()
    for run_id, run_df in df.groupby("run"):
        a_sm = smooth(run_df["a_cmd_total"], window=50)
        plt.plot(run_df["step"], a_sm, label=f"Run {run_id}", alpha=0.9)
    plt.xlabel("Step")
    plt.ylabel("Acceleration [m/s^2]")
    plt.title("Smoothed acceleration vs. time across runs")
    plt.legend()
    plt.grid(True)

    # 3) Smoothed time headway vs time for each run
    if "time_headway" in df.columns:
        plt.figure()
        for run_id, run_df in df.groupby("run"):
            h_sm = smooth(run_df["time_headway"], window=50)
            plt.plot(run_df["step"], h_sm, label=f"Run {run_id}", alpha=0.9)
        plt.axhline(1.0, color="r", linestyle="--", label="1 s threshold")
        plt.xlabel("Step")
        plt.ylabel("Time headway [s]")
        plt.title("Smoothed time headway vs. time across runs")
        plt.legend()
        plt.grid(True)

    # 4) Collisions vs time across runs, clamped to 1
    plt.figure()
    for run_id, run_df in df.groupby("run"):
        plt.step(run_df["step"], run_df["collision_cum_clamped"],
                 where="post", label=f"Run {run_id}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative collisions (clamped to 1)")
    plt.title("RL collisions vs. time across runs")
    plt.legend()
    plt.grid(True)

    # 5) Per-run bar plots for clean summary (optional)
    plt.figure()
    plt.bar(run_stats["run"], run_stats["rms_accel"])
    plt.xlabel("Run")
    plt.ylabel("RMS acceleration [m/s^2]")
    plt.title("RMS acceleration per run")
    plt.grid(axis="y")

    plt.figure()
    plt.bar(run_stats["run"], run_stats["collision_events"])
    plt.xlabel("Run")
    plt.ylabel("Collision events")
    plt.title("Collision events per run")
    plt.grid(axis="y")

plt.show()
