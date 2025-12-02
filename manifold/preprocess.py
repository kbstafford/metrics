# preprocess.py (robust)
import os, argparse, numpy as np, h5py
from one.api import ONE

def choose_eid(one, project, prefer_trials=True):
    # try sessions that have spikes (and trials if requested)
    if prefer_trials:
        eids = one.search(project=project, datasets=['spikes.times.npy', '_ibl_trials.intervals.npy'])
        if eids:
            return eids[0]
    eids = one.search(project=project, datasets=['spikes.times.npy'])
    if eids:
        return eids[0]
    # last resort: any session
    eids = one.search(project=project)
    if not eids:
        raise RuntimeError("No sessions found in this project.")
    return eids[0]


def load_trials(one, eid, want=None):
    """
    want: 'intervals' or 'goCue_times' (None -> return empty dict)
    Returns only the requested array to avoid huge downloads.
    """
    if not want:
        return {}
    try:
        obj = one.load_object(eid, 'trials', attribute=[want])
        arr = obj.get(want, None)
        return {want: arr} if arr is not None else {}
    except Exception:
        return {}


def download_spikes(one, eid, collection=None):
    # Try a few common collections and only fetch the tiny arrays we need
    collections = [collection] if collection else [
        'alf/probe00/pykilosort', 'alf/probe00', 'alf', None
    ]
    for coll in collections:
        try:
            obj = one.load_object(
                eid, 'spikes',
                collection=coll,
                attribute=['times', 'clusters']  # <-- key line
            )
            st, sc = obj.get('times'), obj.get('clusters')
            if st is not None and sc is not None:
                return st, sc
        except Exception:
            continue
    raise RuntimeError("Couldn't load spikes.times/clusters for this session.")


def build_windows(trials_dict, mode, t_before, t_after, spike_times, max_seconds=None):
    """
    Returns a list of (trial_idx, t0, t1) tuples.
    mode: 'intervals' | 'goCue' | 'session'
    """
    if mode == 'intervals' and 'intervals' in trials_dict:
        arr = trials_dict['intervals']
        return [(i, float(t0), float(t1))
                for i, (t0, t1) in enumerate(arr)
                if np.isfinite(t0) and np.isfinite(t1) and t1 > t0]

    if mode == 'goCue' and 'goCue_times' in trials_dict:
        go = trials_dict['goCue_times']
        return [(i, float(t - t_before), float(t + t_after))
                for i, t in enumerate(go) if np.isfinite(t)]

    if mode == 'session':
        t0 = float(np.nanmin(spike_times))
        t1 = float(np.nanmax(spike_times))
        if max_seconds is not None:
            t1 = min(t1, t0 + float(max_seconds))
        return [(0, t0, t1)]

    # nothing found for requested mode
    return []


def bin_matrix(st, sc, neuron_ids, t0, t1, dt):
    edges = np.arange(t0, t1 + dt, dt)
    if edges.size < 2: return None
    centers = edges[:-1] + 0.5*dt
    M = np.empty((centers.size, 1 + 2*len(neuron_ids)), float)
    M[:,0] = centers
    mask = (st >= t0) & (st < t1)
    st_w, sc_w = st[mask], sc[mask]
    for i, clu in enumerate(neuron_ids):
        counts, _ = np.histogram(st_w[sc_w==clu], bins=edges)
        M[:,1+2*i]   = counts.astype(float)
        M[:,1+2*i+1] = np.nan
    return M

def write_h5(path, trial_mats):
    if os.path.exists(path): os.remove(path)
    with h5py.File(path,'w') as f:
        for idx, M in trial_mats:
            g = f.create_group(f"trial_{idx:04d}")
            g.create_dataset('signals', data=M, compression='gzip')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='brainwide')
    ap.add_argument('--eid', default=None)
    ap.add_argument('--dt', type=float, default=0.001)
    ap.add_argument('--max-seconds', type=float, default=60.0,
                    help='Cap session window length (s) when --align session')
    ap.add_argument('--align', choices=['intervals','goCue','session'], default='intervals')
    ap.add_argument('--t-before', type=float, default=0.5)
    ap.add_argument('--t-after', type=float, default=1.0)
    ap.add_argument('--gt-out', default='ibl_ground_truth.h5')
    ap.add_argument('--sub-out', default='ibl_submission.h5')
    args = ap.parse_args()

    one = ONE(base_url='https://openalyx.internationalbrainlab.org')
    eid = args.eid or choose_eid(one, args.project, prefer_trials=(args.align == 'intervals'))

    print(f"[ONE] Using EID: {eid}", flush=True)
    spikes_t, spikes_c = download_spikes(one, eid)
    print(f"[ONE] Loaded spikes: n_spikes={spikes_t.size}, n_units={np.unique(spikes_c).size}", flush=True)

    want = 'intervals' if args.align == 'intervals' else ('goCue_times' if args.align == 'goCue' else None)
    trials_dict = load_trials(one, eid, want)
    if trials_dict:
        k = next(iter(trials_dict.keys()))
        print(f"[ONE] Loaded trials.{k}: n={len(trials_dict[k])}", flush=True)
    else:
        print("[ONE] No trials field found (using whole-session window).", flush=True)

    mode = args.align
    windows = build_windows(trials_dict, mode, args.t_before, args.t_after,
                            spikes_t, max_seconds=(args.max_seconds if mode == 'session' else None))

    # Optional fallback if none found
    if not windows:
        # fall back to a short session window
        windows = build_windows({}, 'session', args.t_before, args.t_after,
                                spikes_t, max_seconds=args.max_seconds)

    print(f"[prep] Windows: {len(windows)}", flush=True)
    # --- BIN + WRITE (paste this right after the "[prep] Windows:" print) ---
    print(f"[prep] Binning {len(windows)} window(s) at dt={args.dt}s …", flush=True)

    # pick neurons (optionally cap for quick tests)
    neuron_ids = np.unique(spikes_c)
    # e.g., uncomment next line to cap:
    # neuron_ids = neuron_ids[:256]

    mats = []
    for idx, t0, t1 in windows:
        print(f"[prep]  Window {idx}: {t0:.3f}–{t1:.3f}s over {len(neuron_ids)} neurons", flush=True)
        M = bin_matrix(spikes_t, spikes_c, neuron_ids, t0, t1, args.dt)
        if M is not None and M.shape[0] > 0:
            mats.append((idx, M))

    print(f"[prep] built {len(mats)} matrices", flush=True)
    if not mats:
        raise RuntimeError("All windows empty—try a larger dt (e.g., --dt 0.005) or a different EID.")

    import os
    out_gt = os.path.abspath(args.gt_out)
    out_sub = os.path.abspath(args.sub_out)
    os.makedirs(os.path.dirname(out_gt), exist_ok=True)
    os.makedirs(os.path.dirname(out_sub), exist_ok=True)

    print(f"[write] -> {out_gt}", flush=True)
    write_h5(out_gt, mats)
    print(f"[write] -> {out_sub}", flush=True)
    write_h5(out_sub, mats)
    print("[done] HDF5 written", flush=True)

    # Final summary
    n_cols = mats[0][1].shape[1]
    n_neurons = (n_cols - 1) // 2
    print(f"✓ Wrote {out_gt} and {out_sub}")
    print(f"EID={eid} | trials={len(mats)} | neurons={n_neurons} | dt={args.dt}s | align={args.align}")
    # --- END BIN + WRITE ---


if __name__ == '__main__':
    main()
