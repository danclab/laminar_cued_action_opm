import os
import os.path as op
import re
import sys

import mne
import numpy as np
import spm_standalone
from lameg.util import batch
from mne import read_epochs

BASE_DIR = "/home/bonaiuto/laminar_cued_action_opm/OPM-LMN.260209"

def find_runs(meg_dir):
    """Find all R*.tsss.fif files and sort numerically."""
    files = [f for f in os.listdir(meg_dir) if f.startswith("R") and f.endswith(".fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files


def process_subject(sub_id, spm, modality):
    meg_dir = op.join(BASE_DIR, modality, 'proc', sub_id)

    run_paths, run_names = find_runs(meg_dir)
    session_epochs = []

    for epo_path, name in zip(run_paths, run_names):
        run_id = name.split('.')[0]
        cfg = {
            "spm": {
                "meeg": {
                    "convert": {
                        "dataset": np.asarray([epo_path], dtype="object"),
                        "mode": {
                            'epoched': {
                                'usetrials': True
                            }
                        },
                        'outfile': os.path.join(meg_dir, f'spm_{run_id}-epo')
                    }
                }
            }
        }
        batch(cfg, spm_instance=spm)
        epochs = read_epochs(epo_path, verbose=True)
        session_epochs.append(epochs)

    if modality=='squid':
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

        cfg = {
            "spm": {
                "meeg": {
                    "convert": {
                        "dataset": np.asarray([op.join(meg_dir, 'all-epo.fif')], dtype="object"),
                        "mode": {
                            'epoched': {
                                'usetrials': float(1)
                           }
                        },
                        'outfile': op.join(meg_dir, 'spm_all-epo')
                    }
                }
            }
        }
        batch(cfg, spm_instance=spm)

if __name__ == "__main__":
    subj_id = sys.argv[1]
    modality = sys.argv[2]

    print(f"Processing subject: {subj_id}")
    spm = spm_standalone.initialize()
    process_subject(subj_id, spm, modality)
    spm.terminate()