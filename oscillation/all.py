import numpy as np
from psd import band_power_error
from plv import compute_plv
from pac import compute_pac
from morlet_wavelet import morlet_error
from hilbert_spectrum import hilbert_spectrum_error
from itpc import compute_itpc

def evaluate_oscillation_metrics(V_real, V_model, dt,
                                  spike_times, V_trials=None):

    # 1. Band power
    band_errors = band_power_error(V_real, V_model, dt, spike_times)

    # 2. PLV in theta and gamma
    plv_theta = compute_plv(V_real, V_model, dt, spike_times, 4,  8)
    plv_gamma = compute_plv(V_real, V_model, dt, spike_times, 30, 100)

    # 3. ITPC (requires multiple trials)
    if V_trials is not None:
        itpc, t, f = compute_itpc(V_trials, dt, event_times=[0])
    else:
        itpc = None

    # 4. PAC
    MVL, MI, amp_by_phase, phase_centers = compute_pac(
                            V_real, dt, spike_times)

    # 5. Morlet error
    pow_err, phase_err, freqs = morlet_error(
                            V_real, V_model, dt, spike_times)

    # 6. Hilbert spectrum error
    h_err, h_freq_err, h_freqs = hilbert_spectrum_error(
                            V_real, V_model, dt, spike_times)

    print(f"PLV theta:            {plv_theta:.4f}  (1=perfect)")
    print(f"PLV gamma:            {plv_gamma:.4f}  (1=perfect)")
    print(f"PAC MVL:              {MVL:.4f}")
    print(f"PAC MI:               {MI:.4f}")
    print(f"Morlet power error:   {np.mean(pow_err):.4f}")
    print(f"Hilbert spectrum err: {h_err:.4f}")

    for band, vals in band_errors.items():
        print(f"Band power {band:6s}:  "
              f"real={vals['power_real']:.4f}  "
              f"model={vals['power_model']:.4f}  "
              f"rel_err={vals['rel_error']:.4f}")