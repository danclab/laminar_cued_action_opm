import os
import re
import sys
import json
import os.path as op
from os import sep
import numpy as np
from mne import read_epochs, set_log_level
from utilities import files
from autoreject import AutoReject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

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

    run_paths, run_names = find_runs(meg_dir)
    for epo_path, name in zip(run_paths, run_names):
        run_id = name.split('-')[1]
        print("EPO:", epo_path.split(sep)[-1])

        epochs = read_epochs(epo_path, verbose=False, preload=True)
        print("AMOUNT OF EPOCHS:", len(epochs))

        fig = epochs.average().plot(spatial_colors=True, show=False)
        plt.savefig(op.join(qc_folder, f"{run_id}-pre-autorej_erf.png"))
        plt.close("all")

        ar = AutoReject(
            consensus=np.linspace(0, 1.0, 27),
            n_interpolate=np.array([1, 4, 32]),
            thresh_method="bayesian_optimization",
            cv=10,
            n_jobs=-1,
            random_state=42,
            verbose="progressbar"
        )
        ar.fit(epochs)

        ar_fname = op.join(
            qc_folder,
            f"{run_id}-autoreject.h5"
        )
        ar.save(ar_fname, overwrite=True)
        epochs_ar, rej_log = ar.transform(epochs, return_log=True)

        rej_log.plot(show=False)
        plt.savefig(op.join(qc_folder, f"{run_id}-autoreject-log.png"))
        plt.close("all")

        epochs_ar.average().plot(spatial_colors=True, show=False)
        plt.savefig(op.join(qc_folder, f"{run_id}-post-autorej_erf.png"))
        plt.close("all")

        cleaned = op.join(meg_dir, "autoreject-" + epo_path.split(sep)[-1])
        epochs_ar.save(
            cleaned,
            overwrite=True
        )
        print("CLEANED EPOCHS SAVED:", cleaned)


    epo = op.join(meg_dir, 'all-epo.fif')

    print("EPO:", epo.split(sep)[-1])

    epochs = read_epochs(epo, verbose=False, preload=True)
    print("AMOUNT OF EPOCHS:", len(epochs))

    fig = epochs.average().plot(spatial_colors=True, show=False)
    plt.savefig(op.join(qc_folder, "pre-autorej_erf.png"))
    plt.close("all")

    ar = AutoReject(
        consensus=np.linspace(0, 1.0, 27),
        n_interpolate=np.array([1, 4, 32]),
        thresh_method="bayesian_optimization",
        cv=10,
        n_jobs=-1,
        random_state=42,
        verbose="progressbar"
    )
    ar.fit(epochs)

    ar_fname = op.join(
        qc_folder,
        "autoreject.h5"
    )
    ar.save(ar_fname, overwrite=True)
    epochs_ar, rej_log = ar.transform(epochs, return_log=True)

    rej_log.plot(show=False)
    plt.savefig(op.join(qc_folder, "autoreject-log.png"))
    plt.close("all")

    epochs_ar.average().plot(spatial_colors=True, show=False)
    plt.savefig(op.join(qc_folder, "post-autorej_erf.png"))
    plt.close("all")

    cleaned = op.join(meg_dir, "autoreject-" + epo.split(sep)[-1])
    epochs_ar.save(
        cleaned,
        overwrite=True
    )
    print("CLEANED EPOCHS SAVED:", cleaned)


if __name__ == "__main__":
    subj_idx = int(sys.argv[1])
    modality = sys.argv[2]

    subjects = sorted(
        d for d in os.listdir(BASE_DIR)
        if d.startswith("S") and op.isdir(op.join(BASE_DIR, d))
    )

    print(f"Processing subject: {subjects[subj_idx]}")
    process_subject(op.join(BASE_DIR, subjects[subj_idx]), modality)