# manifold/test.py
from pathlib import Path
from .data_loader import load_and_prepare_data

HERE = Path(__file__).resolve().parent         # ...\metrics\manifold
gt_path  = HERE / "ibl_ground_truth.h5"
sub_path = HERE / "ibl_submission.h5"

print("Loading:", gt_path, "|", sub_path)

gt_df, sub_df, gt_trials, sub_trials, spike_cols, mempot_cols, n_gt, n_sub = \
    load_and_prepare_data(str(gt_path), str(sub_path))

print(gt_df.head())
print(f"n_gt={n_gt}, n_sub={n_sub}, #spike_cols={len(spike_cols)}, #mempot_cols={len(mempot_cols)}")
