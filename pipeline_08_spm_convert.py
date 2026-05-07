import os
import os.path as op
import re
import sys

import numpy as np
import spm_standalone
from lameg.util import batch

from utilities import files

BASE_DIR = "/home/common/bonaiuto/laminar_cued_action_opm/derivatives"

def find_runs(meg_dir):
    """Find all R*.tsss.fif files and sort numerically."""
    files = [f for f in os.listdir(meg_dir) if f.startswith("autoreject-R") and f.endswith("-epo.fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files


def process_subject(sub_dir, spm):
    meg_dir = op.join(sub_dir, "squid", "meg")
    qc_folder = op.join(meg_dir, "QC")
    files.make_folder(qc_folder)

    run_paths, run_names = find_runs(meg_dir)
    for epo_path, name in zip(run_paths, run_names):
        run_id = name.split('-')[1]
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
                        'outfile': os.path.join(meg_dir, f'spm_autoreject-{run_id}-epo')
                    }
                }
            }
        }
        batch(cfg, spm_instance=spm)

    cfg = {
        "spm": {
            "meeg": {
                "convert": {
                    "dataset": np.asarray([op.join(meg_dir, 'autoreject-all-epo.fif')], dtype="object"),
                    "mode": {
                        'epoched': {
                            'usetrials': float(1)
                       }
                    },
                    'outfile': op.join(meg_dir, 'spm_autoreject-all-epo')
                }
            }
        }
    }
    batch(cfg, spm_instance=spm)

if __name__ == "__main__":
    subj_idx = int(sys.argv[1])

    subjects = sorted(
        d for d in os.listdir(BASE_DIR)
        if d.startswith("S") and op.isdir(op.join(BASE_DIR, d))
    )

    print(f"Processing subject: {subjects[subj_idx]}")
    spm = spm_standalone.initialize()
    process_subject(op.join(BASE_DIR, subjects[subj_idx]), spm)
    spm.terminate()