import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# --- Load logs from Step 3 and Step 4 ---

baseline = pd.read_csv("CBF_Controller.csv")   # Step 3
rl_log   = pd.read_csv("RL_long_latest.csv")              # Step 4

dt = 0.1  # simulation time step (s), keep in sync with your controllers

# --- 1. Speed over time ---

plt.figure()
plt.plot(baseline["step"], baseline["av_speed"], label="Baseline ACC")
plt.plot(rl_log["step"], rl_log["av_speed"], label="ACC + RL", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Speed [m/s]")
plt.legend()
plt.title("Speed vs. time")
plt.grid(True)

# --- 2. Acceleration and jerk ---

# Baseline: use a_cmd (Step 3)
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

plt.figure()
plt.plot(baseline["step"], a_base, label="Baseline ACC a", alpha=0.8)
plt.plot(rl_log["step"], a_rl, label="ACC + RL a", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()
plt.title("Acceleration vs. time")
plt.grid(True)

plt.figure()
plt.plot(baseline["step"], jerk_base, label="Baseline ACC jerk", alpha=0.8)
plt.plot(rl_log["step"], jerk_rl, label="ACC + RL jerk", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Jerk [m/s^3]")
plt.legend()
plt.title("Jerk vs. time")
plt.grid(True)

# --- 3. Time headway / safety ---

plt.figure()
plt.plot(baseline["step"], baseline["time_headway"], label="Baseline ACC")
plt.plot(rl_log["step"], rl_log["time_headway"], label="ACC + RL", alpha=0.8)
plt.axhline(1.0, color="r", linestyle="--", label="1 s threshold")
plt.xlabel("Step")
plt.ylabel("Time headway [s]")
plt.legend()
plt.title("Time headway vs. time")
plt.grid(True)

# --- 4. Collisions over time ---

if "collision_count" in baseline.columns and "collision_count" in rl_log.columns:
    plt.figure()
    plt.plot(baseline["step"], baseline["collision_count"],
             label="Baseline ACC collisions")
    plt.plot(rl_log["step"], rl_log["collision_count"],
             label="ACC + RL collisions", alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Cumulative collisions")
    plt.legend()
    plt.title("Collisions vs. time")
    plt.grid(True)

# --- 5. Summary comfort & safety metrics ---

def comfort_stats(name: str, df: pd.DataFrame, a: np.ndarray, jerk: np.ndarray):
    print(f"\n=== {name} ===")
    mean_speed = df["av_speed"].mean()
    print(f"Mean speed: {mean_speed:.3f} m/s")
    print(f"RMS accel: {np.sqrt(np.mean(a**2)):.3f} m/s^2")
    print(f"RMS jerk : {np.sqrt(np.mean(jerk**2)):.3f} m/s^3")
    print(f"Max decel (most negative a): {np.min(a):.3f} m/s^2")

    if "collision_count" in df.columns:
        total_collisions = int(df["collision_count"].iloc[-1])
        print(f"Total collisions: {total_collisions}")

    if "time_headway" in df.columns:
        unsafe_steps = int((df["time_headway"] < 1.0).sum())
        print(f"Steps with time_headway < 1 s: {unsafe_steps}")

comfort_stats("Baseline ACC (Step 3)", baseline, a_base, jerk_base)
comfort_stats("ACC + RL (Step 4)", rl_log, a_rl, jerk_rl)

plt.show()
