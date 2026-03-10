# Deconvolution method for extracting post-synaptic potential
import numpy as np

def find_tau(V, dt, t1, t2):
    """
    Find membrane time constant by minimizing flatness
    of deconvolution between t1 and t2 (baseline region)
    """
    best_tau = None
    best_flatness = np.inf

    for tau in range(start, end, step):
        D_trial = deconvolve(V, tau, dt)

        # measure flatness in baseline region
        baseline = D_trial[t1:t2]
        flatness = np.mean(baseline ** 2) / tau

        if flatness < best_flatness:
            best_flatness = flatness
            best_tau = tau

    return best_tau


def deconvolve(V, tau, dt):
    """
    Apply deconvolution: D(t) = tau * dV/dt + V
    """
    dV_dt = np.diff(V) / dt          # numerical derivative
    D = tau * dV_dt + V[:-1]      # add back original voltage
    return D


def reconvolve(D_cropped, tau, dt):
    """
    Reverse the deconvolution to recover isolated PSP shape
    Uses forward Euler integration of: tau * dV/dt = D - V
    """
    V_out = zeros(len(D_cropped))
    for k in range(1, len(D_cropped)):
        V_out[k] = V_out[k-1] + (dt / tau) * (D_cropped[k-1] - V_out[k-1])
    return V_out


def extract_psps(V, dt, t1, t2):
    """
    Full pipeline
    """
    # Step 1 — find membrane time constant
    tau = find_tau(V, dt, t1, t2)

    # Step 2 — deconvolve entire trace
    D = deconvolve(V, tau, dt)

    # Step 3 — find peaks in D (these are the PSP events)
    peaks = find_peaks(D)

    isolated_psps = []

    for peak in peaks:
        # Step 4 — crop window around each peak (-5ms to +15ms)
        D_cropped = crop(D, peak, before=5ms, after=15ms)

        # Step 5 — replace baseline outside crop with mean
        D_cropped = replace_outside_with_mean(D_cropped)

        # Step 6 — reconvolve to recover clean PSP shape
        psp = reconvolve(D_cropped, tau, dt)
        isolated_psps.append(psp)

    # Step 7 — checksum: sum isolated PSPs and compare to original V
    V_reconstructed = sum(isolated_psps)
    error = mean((V_reconstructed - V) ** 2)

    return isolated_psps, error