import os
import re
import sys
import json
import mne
import os.path as op

from extra.tools import dump_the_dict
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

    print("\nFound runs:")
    for r in run_names:
        print(f"  {r}")

    qc_folder = op.join(meg_dir, "QC")
    files.make_folder(qc_folder)

    ica_json = dict()

    for raw_path, name in zip(run_paths, run_names):
        print(f"\nReading {raw_path}")
        run_id = name.split('-')[1]
        eve_path = op.join(
            meg_dir,
            "{}-eve.fif".format(run_id)
        )
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", eve_path)

        raw = mne.io.read_raw_fif(raw_path, verbose=False, preload=False)
        events = mne.read_events(eve_path)

        raw_filtered = raw.copy()
        raw_filtered = raw_filtered.pick_types(meg=True, eeg=False, ref_meg=False)
        raw_filtered.load_data().crop(
            tmin=raw_filtered.times[events[0, 0]]
        )
        raw_filtered.filter(
            l_freq=1.,
            h_freq=60,
            n_jobs=-1
        )

        ica = mne.preprocessing.ICA(
            method="infomax",
            fit_params=dict(extended=True),
            n_components=25,
            max_iter=5000
        )
        ica.fit(raw_filtered)

        ica_name = "{}-ica.fif".format(run_id)

        ica_file = op.join(
            meg_dir,
            ica_name
        )

        ica.save(ica_file, overwrite=True)

        ica_json[ica_name] = []

    ica_json_path = op.join(
        meg_dir,
        "ICA_to_reject.json"
    )
    if not op.exists(ica_json_path):
        dump_the_dict(
            ica_json_path,
            ica_json
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