import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters ---
dt = 0.1  # simulation time step (s) – keep in sync with controller
SMOOTH_WINDOW = 10  # number of steps for moving average; adjust (e.g. 5–20)

# --- Helper functions ---

def smooth_series(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    """
    Simple moving average to make plots less noisy.
    """
    return series.rolling(window=window, center=True).mean()

def add_collision_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-collision events and cumulative events from a cumulative counter.
    Assumes 'collision_count' is a non-decreasing integer counter.
    """
    if "collision_count" not in df.columns:
        return df

    # Difference of cumulative counter -> increments per step
    events = df["collision_count"].diff().fillna(0)
    # Clip any negative artefacts (should not normally happen)
    events = events.clip(lower=0).astype(int)
    df["collision_event"] = events
    df["collision_event_cum"] = df["collision_event"].cumsum()
    return df

# --- Load logs from Step 3 and Step 4 ---

baseline = pd.read_csv("CBF_Controller.csv")  # Step 3
rl_log = pd.read_csv("RL_long.csv")           # Step 4

baseline = add_collision_events(baseline)
rl_log = add_collision_events(rl_log)

# --- 1. Speed over time (smoothed) ---

plt.figure()
plt.plot(baseline["step"],
         smooth_series(baseline["av_speed"]),
         label="Baseline ACC")
plt.plot(rl_log["step"],
         smooth_series(rl_log["av_speed"]),
         label="ACC + RL", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Speed [m/s]")
plt.legend()
plt.title("Speed vs. time (smoothed)")
plt.grid(True)

# --- 2. Acceleration and jerk (smoothed) ---

# Baseline: use a_cmd_total if available, otherwise a_cmd
if "a_cmd_total" in baseline.columns:
    a_base = baseline["a_cmd_total"].values
else:
    a_base = baseline["a_cmd"].values

# RL: use a_cmd_total (ACC + RL)
a_rl = rl_log["a_cmd_total"].values

# Compute jerk if needed
jerk_base = np.diff(a_base) / dt
jerk_base = np.concatenate([[0.0], jerk_base])

if "jerk" in rl_log.columns:
    jerk_rl = rl_log["jerk"].values
else:
    jerk_rl = np.diff(a_rl) / dt
    jerk_rl = np.concatenate([[0.0], jerk_rl])

# Acceleration plot
plt.figure()
plt.plot(baseline["step"],
         smooth_series(pd.Series(a_base)),
         label="Baseline ACC a", alpha=0.8)
plt.plot(rl_log["step"],
         smooth_series(pd.Series(a_rl)),
         label="ACC + RL a", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()
plt.title("Acceleration vs. time (smoothed)")
plt.grid(True)

# Jerk plot
plt.figure()
plt.plot(baseline["step"],
         smooth_series(pd.Series(jerk_base)),
         label="Baseline ACC jerk", alpha=0.8)
plt.plot(rl_log["step"],
         smooth_series(pd.Series(jerk_rl)),
         label="ACC + RL jerk", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Jerk [m/s^3]")
plt.legend()
plt.title("Jerk vs. time (smoothed)")
plt.grid(True)

# --- 3. Time headway / safety (smoothed) ---

plt.figure()
plt.plot(baseline["step"],
         smooth_series(baseline["time_headway"]),
         label="Baseline ACC")
plt.plot(rl_log["step"],
         smooth_series(rl_log["time_headway"]),
         label="ACC + RL", alpha=0.8)
plt.axhline(1.0, color="r", linestyle="--", label="1 s threshold")
plt.xlabel("Step")
plt.ylabel("Time headway [s]")
plt.legend()
plt.title("Time headway vs. time (smoothed)")
plt.grid(True)

# --- 4. Collisions over time (events) ---

if "collision_event_cum" in baseline.columns and "collision_event_cum" in rl_log.columns:
    plt.figure()
    plt.step(baseline["step"], baseline["collision_event_cum"],
             where="post", label="Baseline ACC collisions")
    plt.step(rl_log["step"], rl_log["collision_event_cum"],
             where="post", label="ACC + RL collisions", alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Number of collision events")
    plt.legend()
    plt.title("Cumulative collision events vs. time")
    plt.grid(True)

# --- 5. Summary comfort & safety metrics ---

def comfort_stats(name: str,
                  df: pd.DataFrame,
                  a: np.ndarray,
                  jerk: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    mean_speed = df["av_speed"].mean()
    print(f"Mean speed: {mean_speed:.3f} m/s")
    print(f"RMS accel: {np.sqrt(np.mean(a**2)):.3f} m/s^2")
    print(f"RMS jerk : {np.sqrt(np.mean(jerk**2)):.3f} m/s^3")
    print(f"Max decel (most negative a): {np.min(a):.3f} m/s^2")

    if "collision_event" in df.columns:
        total_collisions = int(df["collision_event"].sum())
        print(f"Total collision events: {total_collisions}")

    if "time_headway" in df.columns:
        unsafe_steps = int((df["time_headway"] < 1.0).sum())
        print(f"Steps with time_headway < 1 s: {unsafe_steps}")

comfort_stats("Baseline ACC (Step 3)", baseline, a_base, jerk_base)
comfort_stats("ACC + RL (Step 4)", rl_log, a_rl, jerk_rl)

plt.show()
