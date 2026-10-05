"""Synthetic acoustic plants and offline calibration helpers."""
from __future__ import annotations

import numpy as np
from scipy import linalg, signal


def ls_deconvolve(target, divisor, n_taps, ridge=1e-6):
    """Ridge least-squares FIR ``m`` such that ``divisor * m ~= target``.

    Replaces ``scipy.signal.deconvolve``, which returns only
    ``len(target)-len(divisor)+1`` taps (a single tap for equal lengths) and so
    is not a valid observation-filter design.
    """
    divisor, target = np.asarray(divisor, float), np.asarray(target, float)
    rows = len(divisor) + n_taps - 1
    col = np.r_[divisor, np.zeros(rows - len(divisor))]
    A = linalg.toeplitz(col, np.r_[divisor[0], np.zeros(n_taps - 1)])
    b = np.r_[target, np.zeros(max(0, rows - len(target)))][:rows]
    return np.linalg.solve(A.T @ A + ridge * np.eye(n_taps), A.T @ b)


def make_synthetic_plants(sec_len=32, extra_primary_delay=8):
    """Low-pass FIR plants with realistic delays; returns a dict of arrays."""
    lp = signal.firwin(sec_len, 0.25)
    p_m = np.r_[np.zeros(extra_primary_delay), lp] * 0.8   # noise -> physical mic
    p_v = np.r_[np.zeros(2), p_m] * 0.95                    # + flight lag to eardrum
    s_m = signal.firwin(sec_len, 0.20) * 0.5                # driver -> physical mic
    s_v_0 = np.r_[0.0, s_m][:sec_len] * 0.9                 # driver -> eardrum, 0 deg
    s_v_45 = np.r_[0.0, 0.0, s_m][:sec_len] * 0.75          # lag + damping at 45 deg
    m_v = ls_deconvolve(p_v, p_m, sec_len)
    return dict(p_m=p_m, p_v=p_v, s_m=s_m, s_v_0deg=s_v_0, s_v_45deg=s_v_45, m_v=m_v)
