import numpy as np 
def hilbert_spectrum(V, dt, spike_times,
                     f_low=1, f_high=100,
                     n_freq_bins=50, excl_ms=5):
    """
    Full time-frequency decomposition via Hilbert transform.
    Unlike FFT which assumes stationarity, Hilbert spectrum
    captures how instantaneous frequency and amplitude
    change over time.

    Uses Empirical Mode Decomposition (EMD) approach:
    1. Decompose signal into Intrinsic Mode Functions (IMFs)
    2. Apply Hilbert transform to each IMF
    3. Get instantaneous frequency and amplitude for each IMF
    4. Combine into full spectrum
    """
    mask = build_spike_mask(len(V), spike_times, excl_ms, excl_ms, dt)
    V_masked = V[mask]
    n_times  = len(V_masked)

    # Step 1 — EMD: extract IMFs via sifting process
    imfs = empirical_mode_decomposition(V_masked)

    n_times_out  = n_times
    freq_axis    = np.linspace(f_low, f_high, n_freq_bins)
    time_axis    = np.arange(n_times) * dt
    H_spectrum   = np.zeros((n_freq_bins, n_times_out))

    for imf in imfs:
        # Step 2 — Hilbert transform each IMF
        analytic   = compute_analytic_signal(imf)
        inst_amp   = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))

        # Step 3 — instantaneous frequency = d(phase)/dt
        inst_freq  = np.diff(inst_phase) / (2 * np.pi * dt)
        inst_freq  = np.append(inst_freq, inst_freq[-1])  # pad to same length

        # Step 4 — bin into frequency axis
        for t in range(n_times_out):
            f = inst_freq[t]
            if f_low <= f <= f_high:
                f_idx = int((f - f_low) / (f_high - f_low) * (n_freq_bins - 1))
                H_spectrum[f_idx, t] += inst_amp[t] ** 2

    return H_spectrum, time_axis, freq_axis


def empirical_mode_decomposition(V, max_imfs=8, max_sift=10):
    """
    Extract Intrinsic Mode Functions from signal.
    Each IMF is a narrowband oscillation at a different timescale.
    Sifting process:
    1. Find all local maxima and minima
    2. Interpolate upper and lower envelopes
    3. Subtract mean envelope from signal
    4. Repeat until IMF criteria met
    5. Subtract IMF from residual and repeat
    """
    residual = V.copy()
    imfs     = []

    for _ in range(max_imfs):
        h = residual.copy()

        for _ in range(max_sift):
            # Step 1 — find local maxima and minima
            maxima_idx = find_local_maxima(h)
            minima_idx = find_local_minima(h)

            if len(maxima_idx) < 3 or len(minima_idx) < 3:
                break

            # Step 2 — cubic spline envelopes
            upper_env = cubic_spline(maxima_idx, h[maxima_idx], len(h))
            lower_env = cubic_spline(minima_idx, h[minima_idx], len(h))

            # Step 3 — mean envelope
            mean_env = (upper_env + lower_env) / 2

            # Step 4 — subtract mean
            h = h - mean_env

            # Step 5 — check stopping criterion
            # IMF criteria: zero crossings and extrema differ by at most 1
            n_zeros   = count_zero_crossings(h)
            n_extrema = len(find_local_maxima(h)) + len(find_local_minima(h))
            if abs(n_zeros - n_extrema) <= 1:
                break

        imfs.append(h)
        residual = residual - h

        # stop if residual is monotonic
        if is_monotonic(residual):
            break

    return imfs


def hilbert_spectrum_error(V_real, V_model, dt, spike_times):
    """
    Compare Hilbert spectra of real vs model.
    """
    H_real,  t, freqs = hilbert_spectrum(V_real,  dt, spike_times)
    H_model, _, _     = hilbert_spectrum(V_model, dt, spike_times)

    # error at each frequency band
    freq_error = np.sqrt(np.mean((H_real - H_model)**2, axis=1))

    # total error
    total_error = np.sqrt(np.mean((H_real - H_model)**2))

    return total_error, freq_error, freqs
