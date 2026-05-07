import os
import re
import sys
import json
import mne
import os.path as op
import numpy as np

from utilities import files

BASE_DIR = "/home/common/bonaiuto/laminar_cued_action_opm/derivatives"

def find_runs(meg_dir):
    """Find all R*.tsss.fif files and sort numerically."""
    files = [f for f in os.listdir(meg_dir) if f.startswith("zapline") and f.endswith("-raw.fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files



def process_subject(sub_dir, modality):
    meg_dir = op.join(sub_dir, modality, "meg")
    run_paths, run_names = find_runs(meg_dir)

    if len(run_paths) == 0:
        raise RuntimeError(f"No R*.tsss.fif files found in {meg_dir}")

    print("\nFound runs:")
    for r in run_names:
        print(f"  {r}")

    qc_folder = op.join(meg_dir, "QC")
    files.make_folder(qc_folder)

    ica_json_file = op.join(
        meg_dir,
        "ICA_to_reject.json"
    )

    with open(ica_json_file) as ica_file:
        ica_files = json.load(ica_file)

    ica_keys = list(ica_files.keys())
    ica_keys.sort()

    session_epochs=[]
    for raw_path, name, ica_key in zip(run_paths, run_names, ica_keys):
        print(f"\nReading {raw_path}")
        run_id = name.split('-')[1]
        eve_path = op.join(
            meg_dir,
            "{}-eve.fif".format(run_id)
        )
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", eve_path)

        ica_path = op.join(
            meg_dir,
            ica_key
        )

        print("INPUT RAW FILE:", raw_path)
        print("INPUT EVENT FILE:", eve_path)
        print("INPUT ICA FILE:", ica_path)

        ica_exc = ica_files[ica_key]

        events = mne.read_events(eve_path)

        ica = mne.preprocessing.read_ica(
            ica_path,
            verbose=False
        )

        raw = mne.io.read_raw_fif(
            raw_path,
            verbose=False,
            preload=True
        )

        raw = ica.apply(
            raw,
            exclude=ica_exc,
            verbose=False
        )
        raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)

        epoch = mne.Epochs(
            raw,
            events,
            tmin=-1,
            tmax=3,
            baseline=None,
            #detrend=1,
            preload=True,
            verbose=True
        )
        epoch_path = op.join(
            meg_dir,
            f"{run_id}-epo.fif"
        )

        epoch.save(
            epoch_path,
            fmt="double",
            overwrite=True,
            verbose=False,
        )

        session_epochs.append(epoch)

    all_epochs = mne.concatenate_epochs(session_epochs, on_mismatch="warn", verbose=False)
    epoch_path = op.join(
        meg_dir,
        "all-epo.fif"
    )

    all_epochs.save(
        epoch_path,
        fmt="double",
        overwrite=True,
        verbose=False,
    )


if __name__ == "__main__":
    subj_idx = int(sys.argv[1])
    modality = sys.argv[2]

    subjects = sorted(
        d for d in os.listdir(BASE_DIR)
        if d.startswith("S") and op.isdir(op.join(BASE_DIR, d))
    )

    print(f"Processing subject: {subjects[subj_idx]}")
    process_subject(op.join(BASE_DIR, subjects[subj_idx]), modality)