import numpy as np
from scipy import signal

from savs_anc import SAVS_ANC_System, StreamingSpeechSeparator, ls_deconvolve, run_simulation
from savs_anc.plants import make_synthetic_plants


def test_ls_deconvolve_recovers_known_filter():
    rng = np.random.default_rng(0)
    p_m = signal.firwin(32, 0.25)
    true_m = np.zeros(16)
    true_m[3] = 0.95
    p_v = np.convolve(p_m, true_m)
    m = ls_deconvolve(p_v, p_m, 16)
    assert np.allclose(np.convolve(p_m, m)[: len(p_v)], p_v, atol=1e-3)


def test_interpolation_endpoints_and_midpoint():
    anc = SAVS_ANC_System(sec_path_len=4)
    a, b = np.ones(4), 3 * np.ones(4)
    anc.set_offline_paths(np.zeros(4), a, b, np.zeros(4))
    assert np.allclose(anc.interpolate_secondary_path(0), a)
    assert np.allclose(anc.interpolate_secondary_path(45), b)
    assert np.allclose(anc.interpolate_secondary_path(22.5), 2 * np.ones(4))
    assert np.allclose(anc.interpolate_secondary_path(90), b)   # clamped


def test_separator_mask_high_for_speech_band_low_for_drone():
    fs = 16000
    t = np.arange(fs) / fs
    sep = StreamingSpeechSeparator(fs=fs)
    for x in 0.5 * np.sin(2 * np.pi * 100 * t):
        sep.step(x, x)
    assert sep.mask < 0.1
    sep.reset()
    for x in 0.5 * np.sin(2 * np.pi * 1500 * t):
        sep.step(x, x)
    assert sep.mask > 0.9


def test_leaky_update_does_not_blow_up_with_zero_input():
    p = make_synthetic_plants(32)
    anc = SAVS_ANC_System(64, 32, mu=0.02, nu=1e-4, beta=1e-3)
    anc.set_offline_paths(p["s_m"], p["s_v_0deg"], p["s_v_45deg"], p["m_v"])
    anc.w[:] = 0.1
    for _ in range(5000):
        anc.process_sample(0.0, 0.0, 0.0, 0.0)
    assert np.all(np.isfinite(anc.w)) and np.linalg.norm(anc.w) < 0.1 * np.sqrt(64)


def test_simulation_cancels_noise_at_virtual_eardrum():
    m = run_simulation().metrics
    assert m["virtual_atten_db"] > 15
    assert m["virtual_atten_after_turn_db"] > 15
    assert m["speech_isolation_corr"] > 0.5


def test_head_tracking_helps_after_turn():
    on = run_simulation(track_head=True).metrics["virtual_atten_after_turn_db"]
    off = run_simulation(track_head=False).metrics["virtual_atten_after_turn_db"]
    assert on > off + 3
