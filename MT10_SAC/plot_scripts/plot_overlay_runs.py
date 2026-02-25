import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def find_latest_event_file(path: str) -> str:
    """
    Accepts either:
      - a directory that contains event files (possibly nested)
      - a direct path to an events.out.tfevents.* file
    Returns the latest event file by mtime.
    """
    if os.path.isfile(path) and os.path.basename(path).startswith("events.out.tfevents."):
        return path

    candidates = glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under: {path}")
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def load_scalar_from_event(event_file: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for a single tag from one event file."""
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    if tag not in available:
        raise KeyError(f"Tag '{tag}' not found in {event_file}. Available tags: {sorted(list(available))[:30]} ...")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    vals  = np.array([e.value for e in events], dtype=np.float64)

    # sort + de-dup
    order = np.argsort(steps)
    steps = steps[order]
    vals  = vals[order]
    uniq_steps, idx = np.unique(steps, return_index=True)
    uniq_vals = vals[idx]
    return uniq_steps, uniq_vals


def default_label(path: str) -> str:
    # use last folder name if directory, else use parent folder of event file
    if os.path.isdir(path):
        return os.path.basename(os.path.normpath(path))
    return os.path.basename(os.path.dirname(os.path.normpath(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of run log directories OR direct event file paths."
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional list of labels (same length as --inputs). If omitted, folder names are used."
    )
    parser.add_argument(
        "--tag",
        default="task/ep_success_rate_mean",
        help="Scalar tag to plot (default: task/ep_success_rate_mean)."
    )
    parser.add_argument(
        "--outdir",
        default="report_figures",
        help="Output directory."
    )
    parser.add_argument(
        "--outfile",
        default="overlay_mean_success.png",
        help="Output filename (png)."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.labels is not None and len(args.labels) > 0:
        if len(args.labels) != len(args.inputs):
            raise ValueError(f"--labels must have same length as --inputs ({len(args.inputs)}), got {len(args.labels)}")
        labels = args.labels
    else:
        labels = [default_label(p) for p in args.inputs]

    fig, ax = plt.subplots()
    ax.set_title(f"Overlay: {args.tag}")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Value")

    for path, label in zip(args.inputs, labels):
        event_file = find_latest_event_file(path)
        steps, vals = load_scalar_from_event(event_file, args.tag)
        ax.plot(steps, vals, label=label)

    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(args.outdir, args.outfile)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Saved overlay plot to: {outpath}")


if __name__ == "__main__":
    main()
