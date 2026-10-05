"""SAVS-ANC controller: virtual sensing (RMM/IMC) + normalized leaky FxLMS."""
from __future__ import annotations

import numpy as np

from .speech import StreamingSpeechSeparator


class SAVS_ANC_System:
    """Spatially Adaptive Virtual-Sensing Active Noise Cancellation.

    Per sample the controller:
      1. interpolates the virtual secondary path S_v(z, theta) from head yaw,
      2. isolates speech and purifies the noise reference,
      3. estimates the disturbance at the virtual eardrum from the physical
         error mic via the observation filter M_v(z) ~= P_v(z)/P_m(z),
      4. synthesises anti-noise and adapts with normalized leaky FxLMS.
    """

    def __init__(self, filter_len=64, sec_path_len=32, fs=16000,
                 mu=0.005, nu=1e-5, beta=1e-4, separator=None):
        self.fs, self.L, self.M = fs, filter_len, sec_path_len
        self.mu, self.nu, self.beta = mu, nu, beta

        self.w = np.zeros(self.L)
        self.x_buf = np.zeros(self.L)
        self.x_filt_buf = np.zeros(self.L)
        self.y_buf = np.zeros(self.M)
        self.dm_buf = np.zeros(self.M)

        self.separator = separator or StreamingSpeechSeparator(fs=fs)

        self.s_m_hat = np.zeros(self.M)
        self.s_v_lut = {}
        self.m_v = np.zeros(self.M)

    def set_offline_paths(self, s_m_hat, s_v_0deg, s_v_45deg, m_v):
        """Load offline-calibrated impulse responses (zero-padded/truncated to M taps)."""
        def fit(h):
            h = np.asarray(h, float)
            return np.pad(h, (0, max(0, self.M - len(h))))[: self.M]
        self.s_m_hat = fit(s_m_hat)
        self.s_v_lut[0.0] = fit(s_v_0deg)
        self.s_v_lut[45.0] = fit(s_v_45deg)
        self.m_v = fit(m_v)

    def interpolate_secondary_path(self, head_yaw_deg):
        """Linear interpolation of S_v between the 0 deg and 45 deg calibrations."""
        alpha = float(np.clip(head_yaw_deg, 0.0, 45.0)) / 45.0
        return (1.0 - alpha) * self.s_v_lut[0.0] + alpha * self.s_v_lut[45.0]

    def process_sample(self, x_mic1_n, x_mic2_n, e_m_n, head_yaw_deg,
                       ext_audio_n=0.0, adapt=True, track_head=True):
        """One causal processing cycle.

        Returns ``(y_drive, y_anc, e_v_hat, speech)``.
        ``track_head=False`` freezes S_v at 0 deg (ablation).
        """
        s_v_hat = self.interpolate_secondary_path(head_yaw_deg if track_head else 0.0)

        # Broadside target (0 deg) => zero steering delay, mic2 used directly.
        x_anc_n, s_voice_n = self.separator.step(x_mic1_n, x_mic2_n)
        self.x_buf[1:] = self.x_buf[:-1]
        self.x_buf[0] = x_anc_n

        # Virtual sensing (IMC): strip our own anti-noise from the physical mic.
        y_m_est = self.s_m_hat @ self.y_buf
        d_m_hat = e_m_n - y_m_est
        self.dm_buf[1:] = self.dm_buf[:-1]
        self.dm_buf[0] = d_m_hat
        d_v_hat = self.m_v @ self.dm_buf
        y_v_est = s_v_hat @ self.y_buf
        e_v_hat = d_v_hat + y_v_est

        # Anti-noise synthesis.
        y_anc_n = float(self.w @ self.x_buf)
        self.y_buf[1:] = self.y_buf[:-1]
        self.y_buf[0] = y_anc_n

        # Filtered-x and normalized leaky FxLMS update.
        self.x_filt_buf[1:] = self.x_filt_buf[:-1]
        self.x_filt_buf[0] = s_v_hat @ self.x_buf[: self.M]
        if adapt:
            norm = self.x_filt_buf @ self.x_filt_buf + self.beta
            self.w = (1.0 - self.nu) * self.w - (self.mu / norm) * e_v_hat * self.x_filt_buf

        return y_anc_n + s_voice_n + ext_audio_n, y_anc_n, e_v_hat, s_voice_n
