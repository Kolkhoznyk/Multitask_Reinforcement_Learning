import argparse

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_scalar_events(logdir: str, tag: str) -> list:
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    return ea.Scalars(tag)


def summarize(logdir: str, tag: str, window: int = 50) -> None:
    events = load_scalar_events(logdir, tag)
    if not events:
        print(f"No events found for tag '{tag}' in {logdir}")
        return

    vals = np.array([e.value for e in events])

    print(f"Final value @ step {events[-1].step}: {events[-1].value:.4f}")

    # average over the last `window` logged points to smooth out noise
    tail = vals[-window:]
    print(f"Mean (last {len(tail)} points): {tail.mean():.4f} +/- {tail.std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print summary stats from a TensorBoard log.")
    parser.add_argument("--logdir", default="metaworld_logs/baseline/seed42_4", help="Path to TB event dir.")
    parser.add_argument("--tag", default="task/ep_success_rate_mean", help="Scalar tag to read.")
    parser.add_argument("--window", type=int, default=50, help="Number of trailing points for the rolling average.")
    args = parser.parse_args()

    summarize(args.logdir, args.tag, args.window)


if __name__ == "__main__":
    main()
