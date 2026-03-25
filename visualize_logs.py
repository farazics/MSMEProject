import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import imread

CSV_PATH = "adv_log.csv"
IMG_DIR = "adv_images"


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found")
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"{IMG_DIR} directory not found")

    df = pd.read_csv(CSV_PATH)

    # Basic sanity
    if "cam_frame" not in df.columns:
        raise ValueError("CSV must contain a 'cam_frame' column")

    xs = df["x_av"].values
    ys = df["y_av"].values
    ts = df["t"].values
    frames = df["cam_frame"].values

    # Matplotlib figure with two panes: trajectory + image
    fig, (ax_traj, ax_img) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1]}
    )
    plt.subplots_adjust(bottom=0.15)

    # Trajectory (full path)
    ax_traj.plot(xs, ys, "k-", alpha=0.4, label="trajectory")
    # Marker for current position
    current_point, = ax_traj.plot(xs[0], ys[0], "ro", label="current")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_title("Ego trajectory")
    ax_traj.axis("equal")
    ax_traj.legend(loc="best")

    # Initial image
    img_ax = ax_img.imshow(
        load_image(frames[0]),
        aspect="equal"
    )
    ax_img.axis("off")
    ax_img.set_title(f"Frame {frames[0]}")

    # Slider for time index
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="index",
        valmin=0,
        valmax=len(df) - 1,
        valinit=0,
        valstep=1
    )

    def update(idx):
        idx = int(slider.val)

        # Update trajectory marker
        current_point.set_data(xs[idx], ys[idx])

        # Update image (if file exists)
        frame_id = int(frames[idx])
        img_path = os.path.join(IMG_DIR, f"{frame_id}.png")
        if os.path.exists(img_path) and frame_id >= 0:
            img_ax.set_data(imread(img_path))
            ax_img.set_title(f"Frame {frame_id}")
        else:
            # Blank if image missing
            img_ax.set_data(
                0 * img_ax.get_array()
            )
            ax_img.set_title(f"Frame {frame_id} (missing)")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def load_image(frame_id):
    """
    Helper to load first image (or a blank if missing).
    """
    img_path = os.path.join(IMG_DIR, f"{int(frame_id)}.png")
    if os.path.exists(img_path):
        return imread(img_path)
    else:
        # Minimal blank image fallback
        import numpy as np
        return np.zeros((600, 800, 3), dtype=np.uint8)


if __name__ == "__main__":
    main()
