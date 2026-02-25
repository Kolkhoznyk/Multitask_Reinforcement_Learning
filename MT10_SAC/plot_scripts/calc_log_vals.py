from tensorboard.backend.event_processing import event_accumulator
import numpy as np

logdir = "metaworld_logs/baseline/seed42_4"  # anpassen
tag = "task/ep_success_rate_mean"

ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()

events = ea.Scalars(tag)
final_step = events[-1].step
final_value = events[-1].value

print(f"Final mean success @ step {final_step}: {final_value:.4f}")

vals = np.array([e.value for e in events])
final_mean = vals[-50:].mean()   # letzte 50 Logpunkte
final_std  = vals[-50:].std()

print(f"Final mean success (avg last window): {final_mean:.4f} ± {final_std:.4f}")
