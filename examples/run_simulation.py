"""Run the testbench and save a diagnostic figure to results/."""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from savs_anc.simulation import run_simulation, print_metrics

r = run_simulation(verbose=True)
off = run_simulation(track_head=False)
t = np.arange(len(r.d_v)) / r.fs
smooth = lambda x, w=800: 10 * np.log10(np.convolve(x ** 2, np.ones(w) / w, "same") + 1e-12)

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, smooth(r.d_v), label="Disturbance at eardrum (ANC off)")
ax[0].plot(t, smooth(r.e_v), label="Residual, head tracking ON")
ax[0].plot(t, smooth(off.e_v), label="Residual, head tracking OFF", alpha=0.7)
ax[0].set_ylabel("Power (dB)"); ax[0].legend(); ax[0].set_title("Virtual-eardrum noise power")
ax[1].plot(t, r.speech, label="True speech"); ax[1].plot(t, r.speech_iso, label="Isolated speech", alpha=0.7)
ax[1].set_ylabel("Amplitude"); ax[1].legend(); ax[1].set_title("Speech isolation")
ax[2].plot(t, r.yaw); ax[2].set_ylabel("Yaw (deg)"); ax[2].set_xlabel("Time (s)"); ax[2].set_title("Head trajectory")
for a in ax:
    a.axvspan(1.0, 2.0, color="gold", alpha=0.12)
fig.tight_layout()
out = pathlib.Path(__file__).resolve().parents[1] / "results" / "simulation.png"
fig.savefig(out, dpi=140)
print("Saved", out)
