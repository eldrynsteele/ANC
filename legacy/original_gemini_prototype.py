import numpy as np
import scipy.signal as signal

class SAVS_ANC_System:
    """
    Spatially Adaptive Virtual-Sensing Active Noise Cancellation (SAVS-ANC)
    with Semantic Speech Isolation and Dynamic Head Tracking.
    """
    def __init__(self, filter_len=64, sec_path_len=32, fs=16000, 
                 mu=0.005, nu=1e-5, beta=1e-4):
        self.fs = fs
        self.L = filter_len
        self.M = sec_path_len
        self.mu = mu          # Normalized step size
        self.nu = nu          # Leakage factor
        self.beta = beta      # Regularization factor
        
        # Adaptive ANC control filter weights
        self.w = np.zeros(self.L)
        
        # Circular delay line buffers
        self.x_buf = np.zeros(self.L)
        self.x_filt_buf = np.zeros(self.L)
        self.y_buf = np.zeros(self.M)
        self.dm_buf = np.zeros(self.M)
        
        # MVDR / Directional microphone array configuration
        self.mic_dist = 0.02  # 2 cm separation
        self.c0 = 343.0
        
        # Offline calibrated models: Physical secondary path Sm(z) and Virtual path S0(z), S90(z)
        self.s_m_hat = np.zeros(self.M)
        self.s_v_lut = {}     # Look-up table for head orientations
        self.m_v = np.zeros(self.M)  # Observation filter Mv(z) = Pv(z) / Pm(z)
        
    def set_offline_paths(self, s_m_hat, s_v_0deg, s_v_45deg, m_v):
        """Loads offline-calibrated impulse responses."""
        self.s_m_hat = s_m_hat
        self.s_v_lut[0.0] = s_v_0deg
        self.s_v_lut[45.0] = s_v_45deg
        self.m_v = m_v

    def interpolate_secondary_path(self, head_yaw_deg):
        """
        Dynamically interpolates the virtual secondary path Sv_hat(z, theta) 
        based on the 6-DoF head yaw angle.
        """
        clamped_yaw = np.clip(head_yaw_deg, 0.0, 45.0)
        alpha = clamped_yaw / 45.0
        # Barycentric linear interpolation across angular grid
        s_v_hat = (1.0 - alpha) * self.s_v_lut[0.0] + alpha * self.s_v_lut[45.0]
        return s_v_hat

    def semantic_speech_separator(self, x_mic1, x_mic2, target_angle_deg=0.0):
        """
        Broadside beamforming and energy-envelope VAD masking to isolate 
        conversational speech and purify the ANC reference.
        """
        # Delay-and-sum directional alignment
        delay_sec = (self.mic_dist * np.sin(np.radians(target_angle_deg))) / self.c0
        delay_samples = int(np.round(delay_sec * self.fs))
        
        # Time-aligned beamformed speech candidate
        x2_aligned = x_mic2 if delay_samples == 0 else np.roll(x_mic2, delay_samples)
        s_bf = 0.5 * (x_mic1 + x2_aligned)
        
        # Real-time Voice Activity Detection (envelope-to-variance ratio)
        speech_pwr = np.mean(s_bf**2) + 1e-12
        noise_floor_est = np.mean((x_mic1 - x2_aligned)**2) + 1e-12
        snr_est = speech_pwr / noise_floor_est
        
        # Sigmoidal speech probability mask M_s(n)
        mask = 1.0 / (1.0 + np.exp(-1.5 * (snr_est - 2.0)))
        
        isolated_speech = mask * s_bf
        # Purified ANC reference: subtract speech leakage
        x_anc = x_mic1 - isolated_speech
        return x_anc, isolated_speech

    def process_sample(self, x_mic1_n, x_mic2_n, e_m_n, head_yaw_deg, ext_audio_n=0.0):
        """
        Executes a single-sample processing cycle under causality constraints (<50 us).
        """
        # 1. Update pose-dependent virtual secondary path model
        s_v_hat = self.interpolate_secondary_path(head_yaw_deg)
        
        # 2. Semantic reference separation
        x_anc_n, s_voice_n = self.semantic_speech_separator(
            x_mic1_n, x_mic2_n, target_angle_deg=0.0
        )
        
        # Update reference buffer
        self.x_buf[1:] = self.x_buf[:-1]
        self.x_buf[0] = x_anc_n
        
        # 3. Virtual Sensing (Remote Microphone Method via Internal Model Control)
        # Anti-noise contribution at physical mic: ym_est(n) = s_m_hat * y_buf
        y_m_est = np.dot(self.s_m_hat, self.y_buf[:len(self.s_m_hat)])
        
        # Primary disturbance at physical mic: d_m_hat(n) = e_m(n) - y_m_est(n)
        d_m_hat_n = e_m_n - y_m_est
        
        # Update disturbance buffer
        self.dm_buf[1:] = self.dm_buf[:-1]
        self.dm_buf[0] = d_m_hat_n
        
        # Predict disturbance at virtual eardrum: d_v_hat(n) = m_v * dm_buf
        d_v_hat_n = np.dot(self.m_v, self.dm_buf[:len(self.m_v)])
        
        # Anti-noise contribution at virtual eardrum: y_v_est(n) = s_v_hat * y_buf
        y_v_est = np.dot(s_v_hat, self.y_buf[:len(s_v_hat)])
        
        # Virtual error at tympanic membrane: e_v_hat(n) = d_v_hat(n) + y_v_est(n)
        e_v_hat_n = d_v_hat_n + y_v_est
        
        # 4. Anti-noise synthesis: y_anc(n) = w^T * x
        y_anc_n = np.dot(self.w, self.x_buf)
        
        # Update output buffer (anti-noise)
        self.y_buf[1:] = self.y_buf[:-1]
        self.y_buf[0] = y_anc_n
        
        # 5. Filtered-X generation: x'(n) = s_v_hat * x_buf
        x_filtered_n = np.dot(s_v_hat, self.x_buf[:len(s_v_hat)])
        self.x_filt_buf[1:] = self.x_filt_buf[:-1]
        self.x_filt_buf[0] = x_filtered_n
        
        # 6. Normalized Leaky Weight Update (FxNLMS)
        norm_factor = np.dot(self.x_filt_buf, self.x_filt_buf) + self.beta
        # Gradient descent minimizing estimated virtual eardrum error
        self.w = (1.0 - self.nu) * self.w - (self.mu / norm_factor) * e_v_hat_n * self.x_filt_buf
        
        # 7. Acoustic Output Drive (Combining anti-noise and isolated conversational speech)
        y_drive_n = y_anc_n + s_voice_n + ext_audio_n
        
        return y_drive_n, y_anc_n, e_v_hat_n, s_voice_n


# =====================================================================
# SIMULATION TESTBENCH
# =====================================================================

def run_simulation():
    np.random.seed(42)
    fs = 16000
    duration = 2.5  # seconds
    N = int(fs * duration)
    t = np.arange(N) / fs

    # 1. Synthesize Synthetic Physical Plants (FIR representation)
    sec_len = 32
    # Physical primary path to physical mic (P_m) and virtual eardrum (P_v)
    p_m = signal.firwin(sec_len, [0.05, 0.4], pass_zero=False) * 0.8
    p_v = np.roll(p_m, 2) * 0.95  # Slight acoustic flight lag to the eardrum
    
    # Secondary path from driver to physical mic (S_m) and virtual eardrum (S_v)
    s_m = signal.firwin(sec_len, [0.04, 0.35], pass_zero=False) * 0.5
    s_v_0deg = np.roll(s_m, 1) * 0.9
    s_v_45deg = np.roll(s_m, 2) * 0.75  # Phase lag and damping under head movement
    
    # Auxiliary RMM filter: Mv = Pv / Pm (approximated via least-squares deconvolution)
    m_v, _ = signal.deconvolve(p_v, p_m)
    if len(m_v) < sec_len:
        m_v = np.pad(m_v, (0, sec_len - len(m_v)))
    else:
        m_v = m_v[:sec_len]

    # 2. Synthesize Signals
    # Environmental primary noise: low-frequency engine drone (120 Hz + 240 Hz + colored noise)
    noise_drone = (
        0.7 * np.sin(2 * np.pi * 120 * t) + 
        0.5 * np.sin(2 * np.pi * 240 * t) + 
        0.3 * np.random.randn(N)
    )
    b_col, a_col = signal.butter(4, 600 / (fs / 2), btype='low')
    noise = signal.lfilter(b_col, a_col, noise_drone)
    
    # Desired conversational voice burst between 1.0s and 2.0s
    speech = np.zeros(N)
    idx_start, idx_end = int(1.0 * fs), int(2.0 * fs)
    t_burst = t[idx_start:idx_end]
    # Synthetic speech formants: 500 Hz, 1200 Hz, 2400 Hz modulated
    speech_formants = (
        0.6 * np.sin(2 * np.pi * 500 * t_burst) +
        0.4 * np.sin(2 * np.pi * 1200 * t_burst) +
        0.2 * np.sin(2 * np.pi * 2400 * t_burst)
    )
    speech[idx_start:idx_end] = speech_formants * signal.windows.tukey(len(t_burst), 0.2)

    # Microphone inputs
    # External Mic 1 & 2 capture noise + speech (speech arrives broadside at theta=0)
    x_mic1 = noise + speech
    # Small directional phase delay for non-zero background noise incidence
    x_mic2 = np.roll(noise, 1) + speech  

    # True disturbance arriving at physical mic and eardrum
    d_m = signal.lfilter(p_m, [1.0], noise)
    d_v = signal.lfilter(p_v, [1.0], noise)

    # 3. Initialize Controller
    anc = SAVS_ANC_System(
        filter_len=64, sec_path_len=sec_len, fs=fs, 
        mu=0.02, nu=1e-5, beta=1e-3
    )
    anc.set_offline_paths(s_m_hat=s_m, s_v_0deg=s_v_0deg, s_v_45deg=s_v_45deg, m_v=m_v)

    # Head trajectory: static until 1.5s, then turns 45 degrees
    yaw_trajectory = np.zeros(N)
    yaw_trajectory[int(1.5 * fs):] = 45.0

    # 4. Simulation Execution Buffers
    e_physical_hist = np.zeros(N)
    e_virtual_hist = np.zeros(N)
    speech_isolated_hist = np.zeros(N)
    y_anc_hist = np.zeros(N)

    # Internal state for true physical acoustic convolutions
    sec_state_phys = np.zeros(sec_len - 1)
    sec_state_virt = np.zeros(sec_len - 1)

    print("Running SAVS-ANC real-time loop...")
    for n in range(N):
        current_yaw = yaw_trajectory[n]
        s_v_true = s_v_45deg if current_yaw >= 22.5 else s_v_0deg

        # Read true physical error sensor (residual after acoustic superposition)
        # Previous anti-noise passing through physical plant
        y_prev = y_anc_hist[n - 1] if n > 0 else 0.0
        
        # Acoustic convolution through physical paths
        phys_canc, sec_state_phys = signal.lfilter(s_m, [1.0], [y_prev], zi=sec_state_phys)
        virt_canc, sec_state_virt = signal.lfilter(s_v_true, [1.0], [y_prev], zi=sec_state_virt)
        
        # Acoustic superposition at physical mic and virtual eardrum
        e_m_true = d_m[n] + phys_canc[0]
        e_v_true = d_v[n] + virt_canc[0]
        
        # Execute Controller Sample
        y_drv, y_anc, e_v_est, s_iso = anc.process_sample(
            x_mic1[n], x_mic2[n], e_m_true, current_yaw
        )
        
        # Log signals
        y_anc_hist[n] = y_anc
        e_physical_hist[n] = e_m_true
        e_virtual_hist[n] = e_v_true
        speech_isolated_hist[n] = s_iso

    # 5. Performance Metrology and Evaluation
    # Evaluate steady-state cancellation between 0.5s and 1.0s (noise only, converged)
    eval_idx = slice(int(0.5 * fs), int(1.0 * fs))
    pwr_d_virt = np.mean(d_v[eval_idx]**2)
    pwr_e_virt = np.mean(e_virtual_hist[eval_idx]**2)
    pwr_d_phys = np.mean(d_m[eval_idx]**2)
    pwr_e_phys = np.mean(e_physical_hist[eval_idx]**2)

    attenuation_virtual_db = 10.0 * np.log10(pwr_d_virt / pwr_e_virt)
    attenuation_physical_db = 10.0 * np.log10(pwr_d_phys / pwr_e_phys)

    # Evaluate voice preservation during speech interval (1.2s to 1.8s)
    sp_eval_idx = slice(int(1.2 * fs), int(1.8 * fs))
    speech_corr = np.corrcoef(speech[sp_eval_idx], speech_isolated_hist[sp_eval_idx])[0, 1]

    print("\n--- PERFORMANCE RESULTS ---")
    print(f"Virtual Eardrum Noise Attenuation: {attenuation_virtual_db:.2f} dB")
    print(f"Physical Microphone Attenuation:   {attenuation_physical_db:.2f} dB")
    print(f"Target Speech Isolation Correlation: {speech_corr:.4f}")

if __name__ == "__main__":
    run_simulation()