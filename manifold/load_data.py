import requests
from pathlib import Path
import numpy as np

# Use your existing directory
save_dir = Path(r'C:\Users\kayla\Documents\ccf_datascience\metrics\datasets')

base_url = "https://ibl.flatironinstitute.org/public/churchlandlab/Subjects/CSHL024/2020-09-12/001"

files_to_download = {
    'spikes.times.npy': f"{base_url}/alf/probe00/iblsorter/spikes.times.6dd1c1b7-949f-49a1-931b-53fb7b2c0156.npy",
    'spikes.clusters.npy': f"{base_url}/alf/probe00/iblsorter/spikes.clusters.bb5b1a58-434a-4db7-bd1a-3688be1b1eb6.npy",
}

# Need to find the trial files - they're in the _ibl_trials.table.pqt file
# Let's download the parquet table which contains all trial info
trial_table_url = f"{base_url}/alf/_ibl_trials.table.9ca3f169-7e4c-458c-8083-fecacd97c793.pqt"

print("Downloading files...")

# Download spike files
for filename, url in files_to_download.items():
    filepath = save_dir / filename

    if filepath.exists():
        print(f"✓ {filename} already exists")
        continue

    print(f"Downloading {filename} (this may take a few minutes)...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')

        data = np.load(filepath)
        print(f"\n  ✓ Saved - shape: {data.shape}")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")

# Download trial table
print("\nDownloading trial table...")
try:
    response = requests.get(trial_table_url)
    response.raise_for_status()

    trial_filepath = save_dir / 'trials.table.pqt'
    with open(trial_filepath, 'wb') as f:
        f.write(response.content)
    print(f"  ✓ Saved trials table")
except Exception as e:
    print(f"  ✗ Error: {e}")

print(f"\n✓ All files in {save_dir}")
if not probe_cols:
    raise ValueError(f"No probe collections found! Available: {collections}")

probe_col = probe_cols[0]
print(f"Using collection: {probe_col}")

# --- DOWNLOAD DATA FIRST ---
print("Downloading trial data...")
trials = one.load_object(eid, 'trials', collection='alf', download_only=True)

print("Downloading spike data...")
spikes = one.load_object(
    eid, 'spikes',
    collection=probe_col,
    attribute=['times', 'clusters'],
    download_only=True
)

print("✓ Download complete. Now loading into memory...")

# --- NOW LOAD THE CACHED DATA ---
trials = one.load_object(eid, 'trials', collection='alf')
feedback_type = trials['feedbackType']
feedback_times = trials['feedback_times']

spikes = one.load_object(
    eid, 'spikes',
    collection=probe_col,
    attribute=['times', 'clusters']
)
spike_times = spikes['times']
spike_clusters = spikes['clusters']

print(f"✓ Loaded {len(spike_times)} spikes from {len(np.unique(spike_clusters))} neurons")

# ... rest of your processing code ...

win = 0.5  # seconds after feedback

# (optional) only keep trials with a valid feedback time
valid = ~np.isnan(feedback_times)
feedback_type  = feedback_type[valid]
feedback_times = feedback_times[valid]

n_trials = len(feedback_times)
neuron_ids = np.unique(spike_clusters)
n_neurons = len(neuron_ids)

X = np.zeros((n_trials, n_neurons), dtype=float)

for ti in range(n_trials):
    t0 = feedback_times[ti]
    t1 = t0 + win

    mask = (spike_times >= t0) & (spike_times < t1)
    sp_t = spike_times[mask]
    sp_c = spike_clusters[mask]

    # count spikes per neuron and convert to rate
    for ni, cl in enumerate(neuron_ids):
        X[ti, ni] = np.sum(sp_c == cl) / win  # spikes/s in [0, win] after feedback

bin_size = 0.05  # 50 ms
T = int(win / bin_size)

X_t = np.zeros((n_trials, T, n_neurons), dtype=float)

# REPLACE the slow triple loop with this:
X_t = np.zeros((n_trials, T, n_neurons), dtype=float)

# Pre-compute bin edges
bin_edges = np.linspace(0, win, T + 1)

X_t = np.zeros((n_trials, T, n_neurons), dtype=float)

for ni, cl in enumerate(neuron_ids):
    neuron_spikes = spike_times[spike_clusters == cl]

    for ti in range(n_trials):
        t0 = feedback_times[ti]
        trial_spikes = neuron_spikes[(neuron_spikes >= t0) & (neuron_spikes < t0 + win)]
        relative_times = trial_spikes - t0
        counts, _ = np.histogram(relative_times, bins=bin_edges)
        X_t[ti, :, ni] = counts / bin_size

    # Print progress
    if (ni + 1) % 50 == 0:
        print(f"Processed {ni + 1}/{n_neurons} neurons...")

filename = f'ibl_feedback_timeseries_{eid}.h5'

with h5py.File(filename, 'w') as f:
    # --- main data ---
    f.create_dataset('X_t', data=X_t, compression='gzip')  # (n_trials, T, n_neurons)
    f.create_dataset('feedback_type', data=feedback_type, compression='gzip')
    f.create_dataset('neuron_ids', data=neuron_ids, compression='gzip')

    reward = (feedback_type == 1).astype('int8')
    f.create_dataset('reward', data=reward, compression='gzip')

    # --- metadata ---
    meta = f.create_group('meta')
    meta.attrs['session_eid'] = eid
    meta.attrs['alignment'] = 'feedback_times'
    meta.attrs['bin_size_s'] = float(bin_size)
    meta.attrs['window_length_s'] = float(X_t.shape[1] * bin_size)
    meta.attrs['t0_relative_to_feedback'] = 0.0
    meta.attrs['description'] = (
        'X_t: firing rates time-binned after feedback; '
        'dim = (trials, time_bins, neurons).'
    )

print(f'Saved HDF5 to {filename}')

