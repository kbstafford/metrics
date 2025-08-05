import numpy as np
import glob, os, re

# import data
PATH = r"..\datasets"
BIN_S = 0.050 # 50 ms bins
SCALE = 1e-6
DTYPE = "<i8"

# sort files
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]

# load spike times from all .spk files
spk_files = sorted(glob.glob(os.path.join(PATH, "*.spk")), key=natural_key)
assert spk_files, f"No .tem files found in {PATH}"

spike_trains = []
unit_names = []
for f in spk_files:
    ts_int = np.fromfile(f, dtype=DTYPE)          # raw int64 timestamps
    ts_sec = ts_int.astype(np.float64) * SCALE    # convert to seconds
    if ts_sec.size > 0:
        spike_trains.append(np.sort(ts_sec))      # ensure sorted
        unit_names.append(os.path.basename(f))

# determine analysis window from spikes
t0 = min(st[0] for st in spike_trains if st.size)
t1 = max(st[-1] for st in spike_trains if st.size)
edges = np.arange(t0, t1 + BIN_S, BIN_S)

# build spike-count matrix [time_bins * neurons]
spc_matrix = np.zeros((edges.size - 1, len(spike_trains)), dtype=float)
for i, st in enumerate(spike_trains):
    counts, _ = np.histogram(st, bins=edges)
    spc_matrix[:, i] = counts

# optional: drop silent units
keep = np.where(spc_matrix.sum(axis=0) > 0)[0]
spc_matrix = spc_matrix[:, keep]
unit_names = [unit_names[i] for i in keep]
print(f"Time bins: {spc_matrix.shape[0]}, Units: {spc_matrix.shape[1]}")

# covariance calculation
spc_matrix_c = spc_matrix - spc_matrix.mean(axis=0, keepdims=True)
cov_matrix = (spc_matrix_c.T @ spc_matrix) / (spc_matrix.shape[0] - 1)
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T) # symmetrize
tr, tr2 = np.trace(cov_matrix), np.trace(cov_matrix @ cov_matrix)

# calculate participation ratio
pr = (tr * tr) / tr2 if tr2 > 0 else np.nan

# display participation ratio
print(pr)
