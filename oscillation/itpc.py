def compute_itpc(V_trials, dt, event_times, window_ms=500,
                 f_low=1, f_high=100):
    """
    How consistent is the phase of oscillations across trials
    at each timepoint and frequency?

    V_trials:    shape (n_trials, n_timepoints)
    event_times: stimulus/event onset times in ms
                 used to epoch the data

    ITPC = 1 → phase perfectly consistent across trials
    ITPC = 0 → random phase across trials

    Returns: time x frequency ITPC matrix
    """
    n_trials    = V_trials.shape[0]
    window_bins = int(window_ms / dt)
    freqs       = np.fft.rfftfreq(window_bins, d=dt)
    freq_mask   = (freqs >= f_low) & (freqs <= f_high)
    n_freqs     = np.sum(freq_mask)

    # Step 1 — epoch data around each event
    epochs = []
    for trial in range(n_trials):
        for event_t in event_times:
            event_idx = int(event_t / dt)
            start     = event_idx
            end       = event_idx + window_bins
            if end <= V_trials.shape[1]:
                epochs.append(V_trials[trial, start:end])

    epochs    = np.array(epochs)
    n_epochs  = len(epochs)
    n_times   = window_bins

    # Step 2 — compute phase at each timepoint and frequency
    # using short-time FFT (sliding window)
    ITPC_matrix = np.zeros((n_freqs, n_times))

    for t in range(n_times):
        # small window around each timepoint
        win_start = max(0, t - 16)
        win_end   = min(n_times, t + 16)

        phases = np.zeros((n_epochs, n_freqs), dtype=complex)

        for e in range(n_epochs):
            segment = epochs[e, win_start:win_end]
            fft_seg = np.fft.rfft(segment, n=window_bins)

            # Step 3 — extract phase at each frequency
            phases[e] = np.exp(1j * np.angle(fft_seg[freq_mask]))

        # Step 4 — ITPC = |mean across epochs of unit phase vectors|
        ITPC_matrix[:, t] = np.abs(np.mean(phases, axis=0))

    time_axis = np.arange(n_times) * dt
    freq_axis = freqs[freq_mask]

    return ITPC_matrix, time_axis, freq_axis