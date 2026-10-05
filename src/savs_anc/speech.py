"""Streaming dual-microphone speech separator (beamformer + band-energy VAD)."""
from __future__ import annotations

import numpy as np


class StreamingSpeechSeparator:
    """Sample-by-sample speech isolation for the ANC reference path.

    1. Delay-and-sum beamformer steered at ``target_angle_deg`` (broadside = 0).
    2. Speech-band extraction with a cascaded one-pole high-pass.
    3. Smoothed speech-band / low-band energy ratio -> sigmoid probability mask.

    Only the *speech-band* component is removed from the ANC reference, so the
    low-frequency noise reference is preserved while speech is protected from
    cancellation. State is kept internally, so samples must be fed in order.
    """

    def __init__(self, fs=16000, mic_dist=0.02, c0=343.0, hp_hz=250.0,
                 smooth_hz=40.0, thresh_db=0.0, slope=0.6):
        self.fs, self.mic_dist, self.c0 = fs, mic_dist, c0
        rc = 1.0 / (2 * np.pi * hp_hz)
        self.a_hp = rc / (rc + 1.0 / fs)
        self.alpha = 1.0 - np.exp(-2 * np.pi * smooth_hz / fs)
        self.thresh_db, self.slope = thresh_db, slope
        self.reset()

    def reset(self):
        self._x_prev = np.zeros(2)
        self._y_prev = np.zeros(2)
        self._p_hi = 1e-12
        self._p_lo = 1e-12
        self.mask = 0.0

    def _highpass(self, x):
        for k in range(2):
            y = self.a_hp * (self._y_prev[k] + x - self._x_prev[k])
            self._x_prev[k], self._y_prev[k] = x, y
            x = y
        return x

    def step(self, x1, x2_aligned):
        """Return ``(x_anc, isolated_speech)`` for one pair of aligned samples."""
        s_bf = 0.5 * (x1 + x2_aligned)
        hi = self._highpass(s_bf)
        lo = s_bf - hi
        self._p_hi += self.alpha * (hi * hi - self._p_hi)
        self._p_lo += self.alpha * (lo * lo - self._p_lo)
        ratio_db = 10 * np.log10((self._p_hi + 1e-12) / (self._p_lo + 1e-12))
        self.mask = 1.0 / (1.0 + np.exp(-self.slope * (ratio_db - self.thresh_db)))
        speech = self.mask * hi
        return x1 - speech, speech

    def steering_delay_samples(self, target_angle_deg=0.0):
        delay = self.mic_dist * np.sin(np.radians(target_angle_deg)) / self.c0
        return int(np.round(delay * self.fs))
