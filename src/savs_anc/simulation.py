"""End-to-end SAVS-ANC testbench."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .controller import SAVS_ANC_System
from .plants import make_synthetic_plants


@dataclass
class SimResult:
    fs: int
    noise: np.ndarray
    speech: np.ndarray
    d_v: np.ndarray
    e_v: np.ndarray
    d_m: np.ndarray
    e_m: np.ndarray
    speech_iso: np.ndarray
    yaw: np.ndarray
    metrics: dict


def _atten_db(d, e, sl):
    return 10 * np.log10(np.mean(d[sl] ** 2) / np.mean(e[sl] ** 2))


def run_simulation(duration=2.5, fs=16000, seed=42, track_head=True,
                   mu=0.02, nu=1e-5, beta=1e-3, filter_len=64, verbose=False):
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    sec_len = 32
    pl = make_synthetic_plants(sec_len)

    # Primary noise: 120 Hz + 240 Hz engine drone + low-passed broadband.
    drone = (0.7 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
             + 0.3 * rng.standard_normal(N))
    b, a = signal.butter(4, 600 / (fs / 2), btype="low")
    noise = signal.lfilter(b, a, drone)

    # Conversational speech burst, 1.0-2.0 s (broadside).
    speech = np.zeros(N)
    i0, i1 = int(1.0 * fs), int(2.0 * fs)
    tb = t[i0:i1]
    formants = (0.6 * np.sin(2 * np.pi * 500 * tb) + 0.4 * np.sin(2 * np.pi * 1200 * tb)
                + 0.2 * np.sin(2 * np.pi * 2400 * tb))
    speech[i0:i1] = formants * signal.windows.tukey(len(tb), 0.2)

    x1 = noise + speech
    x2 = np.roll(noise, 1) + speech
    d_m = signal.lfilter(pl["p_m"], [1.0], noise)
    d_v = signal.lfilter(pl["p_v"], [1.0], noise)

    anc = SAVS_ANC_System(filter_len, sec_len, fs, mu, nu, beta)
    anc.set_offline_paths(pl["s_m"], pl["s_v_0deg"], pl["s_v_45deg"], pl["m_v"])

    yaw = np.zeros(N)
    yaw[int(1.5 * fs):] = 45.0

    e_m_h, e_v_h, iso_h, y_h = (np.zeros(N) for _ in range(4))
    zi_m = np.zeros(sec_len - 1)
    zi_v = np.zeros(sec_len - 1)
    for n in range(N):
        s_v_true = pl["s_v_45deg"] if yaw[n] >= 22.5 else pl["s_v_0deg"]
        y_prev = y_h[n - 1] if n else 0.0
        pm, zi_m = signal.lfilter(pl["s_m"], [1.0], [y_prev], zi=zi_m)
        pv, zi_v = signal.lfilter(s_v_true, [1.0], [y_prev], zi=zi_v)
        e_m = d_m[n] + pm[0]
        e_v = d_v[n] + pv[0]
        _, y_anc, _, s_iso = anc.process_sample(x1[n], x2[n], e_m, yaw[n],
                                                track_head=track_head)
        y_h[n], e_m_h[n], e_v_h[n], iso_h[n] = y_anc, e_m, e_v, s_iso

    sl = lambda a_, b_: slice(int(a_ * fs), int(b_ * fs))
    sp = sl(1.2, 1.8)
    metrics = dict(
        virtual_atten_db=_atten_db(d_v, e_v_h, sl(0.5, 1.0)),
        physical_atten_db=_atten_db(d_m, e_m_h, sl(0.5, 1.0)),
        virtual_atten_after_turn_db=_atten_db(d_v, e_v_h, sl(2.0, 2.5)),
        speech_isolation_corr=float(np.corrcoef(speech[sp], iso_h[sp])[0, 1]),
    )
    if verbose:
        print_metrics(metrics)
    return SimResult(fs, noise, speech, d_v, e_v_h, d_m, e_m_h, iso_h, yaw, metrics)


def print_metrics(m):
    print("\n--- PERFORMANCE RESULTS (positive dB = noise reduced) ---")
    print(f"Virtual eardrum attenuation (0.5-1.0 s, converged):   {m['virtual_atten_db']:.2f} dB")
    print(f"Physical mic attenuation    (0.5-1.0 s, converged):   {m['physical_atten_db']:.2f} dB")
    print(f"Virtual attenuation after 45 deg turn (2.0-2.5 s):    {m['virtual_atten_after_turn_db']:.2f} dB")
    print(f"Speech isolation correlation (1.2-1.8 s):             {m['speech_isolation_corr']:.4f}")


def main():
    print("Running SAVS-ANC real-time loop...")
    run_simulation(verbose=True)
    print("\nAblation: head tracking OFF (secondary path fixed at 0 deg)")
    r = run_simulation(track_head=False)
    print(f"Virtual attenuation after head turn: {r.metrics['virtual_atten_after_turn_db']:.2f} dB")


if __name__ == "__main__":
    main()
