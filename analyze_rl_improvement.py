import glob
import os
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

dt = 0.1  # time step

# --- Load all historical RL runs ---

# Assumes files named like RL_long_run1.csv, RL_long_run2.csv, ...
run_files = sorted(glob.glob("RL_long_run*.csv"))

if not run_files:
    raise FileNotFoundError("No RL_long_run*.csv files found in current folder.")

runs = []
for fname in run_files:
    df = pd.read_csv(fname)
    # Extract run index from filename, e.g. RL_long_run3.csv -> 3
    base = os.path.basename(fname)
    run_id_str = "".join(ch for ch in base if ch.isdigit())
    run_id = int(run_id_str) if run_id_str else 0
    df["run_id"] = run_id
    runs.append(df)

# Sort runs by run_id
runs = sorted(runs, key=lambda df: df["run_id"].iloc[0])

# Convenience: colors / labels
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))

# --- Helper to compute accel/jerk and stats ---

def compute_a_and_jerk(df):
    a = df["a_cmd_total"].values
    if "jerk" in df.columns:
        jerk = df["jerk"].values
    else:
        jerk = np.diff(a) / dt
        jerk = np.concatenate([[0.0], jerk])
    return a, jerk

def comfort_stats(name, df, a, jerk):
    print(f"\n=== {name} ===")
    mean_speed = df["av_speed"].mean()
    print(f"Mean speed: {mean_speed:.3f} m/s")
    print(f"RMS accel: {np.sqrt(np.mean(a**2)):.3f} m/s^2")
    print(f"RMS jerk : {np.sqrt(np.mean(jerk**2)):.3f} m/s^3")
    print(f"Max decel (most negative a): {np.min(a):.3f} m/s^2")

    if "collision_count" in df.columns:
        # collision_count is cumulative; total collisions in this run:
        total_collisions = int(df["collision_count"].iloc[-1] -
                               df["collision_count"].iloc[0])
        print(f"Total collisions: {total_collisions}")

    if "time_headway" in df.columns:
        unsafe_steps = int((df["time_headway"] < 1.0).sum())
        print(f"Steps with time_headway < 1 s: {unsafe_steps}")

# --- 1. Speed vs time (all runs) ---

plt.figure()
for df, c in zip(runs, colors):
    run_id = df["run_id"].iloc[0]
    plt.plot(df["step"], df["av_speed"],
             label=f"Run {run_id}", alpha=0.8, color=c)
plt.xlabel("Step")
plt.ylabel("Speed [m/s]")
plt.legend()
plt.title("RL speed vs. time across runs")
plt.grid(True)

# --- 2. Acceleration vs time (all runs) ---

plt.figure()
for df, c in zip(runs, colors):
    run_id = df["run_id"].iloc[0]
    a, _ = compute_a_and_jerk(df)
    plt.plot(df["step"], a,
             label=f"Run {run_id}", alpha=0.6, color=c)
plt.xlabel("Step")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()
plt.title("RL acceleration vs. time across runs")
plt.grid(True)

# --- 3. Jerk vs time (all runs) ---

plt.figure()
for df, c in zip(runs, colors):
    run_id = df["run_id"].iloc[0]
    _, jerk = compute_a_and_jerk(df)
    plt.plot(df["step"], jerk,
             label=f"Run {run_id}", alpha=0.6, color=c)
plt.xlabel("Step")
plt.ylabel("Jerk [m/s^3]")
plt.legend()
plt.title("RL jerk vs. time across runs")
plt.grid(True)

# --- 4. Time headway vs time (all runs) ---

plt.figure()
for df, c in zip(runs, colors):
    run_id = df["run_id"].iloc[0]
    plt.plot(df["step"], df["time_headway"],
             label=f"Run {run_id}", alpha=0.6, color=c)
plt.axhline(1.0, color="r", linestyle="--", label="1 s threshold")
plt.xlabel("Step")
plt.ylabel("Time headway [s]")
plt.legend()
plt.title("RL time headway vs. time across runs")
plt.grid(True)

# --- 5. Collisions vs time (all runs) ---

plt.figure()
for df, c in zip(runs, colors):
    run_id = df["run_id"].iloc[0]
    if "collision_count" in df.columns:
        plt.plot(df["step"], df["collision_count"],
                 label=f"Run {run_id}", alpha=0.8, color=c)
plt.xlabel("Step")
plt.ylabel("Cumulative collisions")
plt.legend()
plt.title("RL collisions vs. time across runs")
plt.grid(True)

# --- 6. Summary stats per run ---

for df in runs:
    run_id = df["run_id"].iloc[0]
    a, jerk = compute_a_and_jerk(df)
    comfort_stats(f"RL run {run_id}", df, a, jerk)

plt.show()
