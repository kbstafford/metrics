import numpy as np
def compute_psd(V, dt, spike_times, excl_ms=5):
    """
    How much power at each frequency in the voltage trace?
    Uses Welch's method for stable estimate.

    Returns: frequencies, power at each frequency
    """
    mask = build_spike_mask(len(V), spike_times, excl_ms, excl_ms, dt)
    V_masked = V[mask]

    # Step 1 — divide trace into overlapping windows
    window_size = int(1.0 / dt)        # 1 second windows
    overlap     = window_size // 2     # 50% overlap

    # Step 2 — apply Hann window to each segment to reduce edge artifacts
    hann_window = np.hanning(window_size)

    psds = []
    start = 0
    while start + window_size < len(V_masked):
        segment = V_masked[start : start + window_size]
        segment = segment * hann_window

        # Step 3 — FFT each segment
        fft_segment = np.fft.rfft(segment)
        psd_segment = (np.abs(fft_segment) ** 2) / window_size
        psds.append(psd_segment)

        start += overlap

    # Step 4 — average across all windows (Welch's method)
    PSD   = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(window_size, d=dt)

    return freqs, PSD