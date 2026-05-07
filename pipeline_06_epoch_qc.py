import os
import re
import sys
import json
from mne import read_epochs, set_log_level
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import os.path as op
from os import sep
from utilities import files
import matplotlib.pylab as plt
from autoreject import compute_thresholds
import numpy as np

set_log_level(verbose=False)

BASE_DIR = "/home/common/bonaiuto/laminar_cued_action_opm/derivatives"

def find_runs(meg_dir):
    """Find all R*.tsss.fif files and sort numerically."""
    files = [f for f in os.listdir(meg_dir) if f.startswith("R") and f.endswith("-epo.fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files



def process_subject(sub_dir, modality):
    meg_dir = op.join(sub_dir, modality, "meg")
    qc_folder = op.join(meg_dir, "QC")
    files.make_folder(qc_folder)

    cmap = colors.ListedColormap(["#FFFFFF", "#CFEEFA", "#FFDE00", "#FF9900", "#FF0000", "#000000"])
    boundaries = [-0.9, -0.1, 1.1, 10, 100, 1000, 10000]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    run_paths, run_names = find_runs(meg_dir)
    for epo_path, name in zip(run_paths, run_names):
        print("INPUT FILE:", epo_path)
        run_id = name.split('-')[1]
        epochs = read_epochs(epo_path, verbose=False)

        ch_thr = compute_thresholds(
            epochs,
            random_state=42,
            method="bayesian_optimization",
            verbose="progressbar",
            n_jobs=-1,
            augment=False
        )
        # save the thresholds in JSON
        ch_list = list(ch_thr.keys())
        ch_list.sort()
        results = np.zeros((len(ch_list), len(epochs)))
        results = results - 1
        for ix, ch in enumerate(ch_list):
            thr = ch_thr[ch]
            ch_tr = epochs.copy().pick_channels([ch]).get_data()
            res = [np.where(ch_tr[i][0] > thr)[0].shape[0] for i in range(len(epochs))]
            res = np.array(res)
            results[ix, :] = res
        img_path = op.join(qc_folder, f"{run_id}-epo-QC.png")
        print(results[:15, :15])
        print(np.min(results), np.max(results))
        print(np.unique(results))

        plt.rcParams.update({'font.size': 5})
        f, ax = plt.subplots(
            figsize=(20, 20),
            dpi=200
        )

        im = ax.imshow(
            results,
            aspect="auto",
            cmap=cmap,
            interpolation="none",
            norm=norm
        )
        f.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
        ax.set_xlabel("Trials")
        ax.set_ylabel("Channels")
        ax.set_xticks(list(range(len(epochs))))
        ax.set_xticklabels([str(i) for i in range(1, len(epochs) + 1)])
        ax.set_yticks(list(range(len(ch_list))))
        ax.set_yticklabels(ch_list)
        ax.grid(color='w', linestyle='-', linewidth=0.2)
        plt.savefig(
            img_path,
            bbox_inches="tight"
        )
        plt.close("all")

    epo = op.join(meg_dir, 'all-epo.fif')
    print("INPUT FILE:", epo)
    epochs = read_epochs(epo, verbose=False)

    ch_thr = compute_thresholds(
        epochs,
        random_state=42,
        method="bayesian_optimization",
        verbose="progressbar",
        n_jobs=-1,
        augment=False
    )
    # save the thresholds in JSON
    ch_list = list(ch_thr.keys())
    ch_list.sort()
    results = np.zeros((len(ch_list), len(epochs)))
    results = results - 1
    for ix, ch in enumerate(ch_list):
        thr = ch_thr[ch]
        ch_tr = epochs.copy().pick_channels([ch]).get_data()
        res = [np.where(ch_tr[i][0] > thr)[0].shape[0] for i in range(len(epochs))]
        res = np.array(res)
        results[ix, :] = res
    img_path = op.join(qc_folder, "epo-QC.png")
    print(results[:15, :15])
    print(np.min(results), np.max(results))
    print(np.unique(results))

    plt.rcParams.update({'font.size': 5})
    f, ax = plt.subplots(
        figsize=(20, 20),
        dpi=200
    )

    im = ax.imshow(
        results,
        aspect="auto",
        cmap=cmap,
        interpolation="none",
        norm=norm
    )
    f.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Channels")
    ax.set_xticks(list(range(len(epochs))))
    ax.set_xticklabels([str(i) for i in range(1, len(epochs) + 1)])
    ax.set_yticks(list(range(len(ch_list))))
    ax.set_yticklabels(ch_list)
    ax.grid(color='w', linestyle='-', linewidth=0.2)
    plt.savefig(
        img_path,
        bbox_inches="tight"
    )
    plt.close("all")


if __name__ == "__main__":
    subj_idx = int(sys.argv[1])
    modality = sys.argv[2]

    subjects = sorted(
        d for d in os.listdir(BASE_DIR)
        if d.startswith("S") and op.isdir(op.join(BASE_DIR, d))
    )

    print(f"Processing subject: {subjects[subj_idx]}")
    process_subject(op.join(BASE_DIR, subjects[subj_idx]), modality)