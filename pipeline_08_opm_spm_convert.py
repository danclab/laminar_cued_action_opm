import csv
import os
import os.path as op
import re
import subprocess
import sys

import mne
import numpy as np
import spm_standalone
import matlab
from lameg.surf import LayerSurfaceSet


BASE_DIR = "/home/bonaiuto/laminar_cued_action_opm/OPM-LMN.260209"


def find_runs(meg_dir):
    files = [f for f in os.listdir(meg_dir) if f.startswith("R") and f.endswith(".fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files


def _mat(flag, orig_mgz):
    out = subprocess.check_output(["mri_info", flag, orig_mgz]).decode().split()
    return np.array([float(x) for x in out]).reshape(4, 4)


def apply_affine_to_point(pt, affine):
    pt1 = np.r_[pt, 1.0]
    return (affine @ pt1)[:3]


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length orientation vector")
    return v / n


def process_subject(sub_id, spm):
    os.environ["SUBJECTS_DIR"] = "/home/bonaiuto/laminar_opm/OPM-LMN.260209/anat/fs/"

    meg_dir = op.join(BASE_DIR, "opm", "proc", sub_id)

    run_paths, run_names = find_runs(meg_dir)

    for epo_path, name in zip(run_paths, run_names):
        run_id = name.split(".")[0]
        data = mne.read_epochs(epo_path)

        trans = mne.read_trans(
            os.path.join(BASE_DIR, "opm", "trans", sub_id, f"{run_id}-trans.fif")
        )
        T_mne = trans["trans"]      # meters
        R_mne = T_mne[:3, :3]

        orig_mgz = os.path.join(
            os.environ["SUBJECTS_DIR"],
            sub_id,
            "mri",
            "orig.mgz",
        )

        # FreeSurfer tkRAS -> scannerRAS (millimeters)
        n_orig = _mat("--vox2ras", orig_mgz)
        t_orig = _mat("--vox2ras-tkr", orig_mgz)
        T_tkras_to_scanner = n_orig @ np.linalg.inv(t_orig)
        R_tkras_to_scanner = T_tkras_to_scanner[:3, :3]

        rows = [["name", "Px", "Py", "Pz", "Ox", "Oy", "Oz"]]

        for chan in data.info["chs"]:
            name = f'{chan["ch_name"]}_opm'
            loc = chan["loc"].copy()

            # position in meters
            pos_m = loc[:3]

            # channel orientation matrix
            first_col = loc[3:6]
            second_col = loc[6:9]
            third_col = loc[9:12]
            rot_mat = np.column_stack([first_col, second_col, third_col])

            # apply trans first in meters
            pos_after_trans_m = apply_affine_to_point(pos_m, T_mne)

            # convert to mm for FreeSurfer affine
            pos_after_trans_mm = pos_after_trans_m * 1000.0

            # apply tkRAS -> scannerRAS in mm
            pos_final_mm = apply_affine_to_point(pos_after_trans_mm, T_tkras_to_scanner)

            # spm_opm_create expects mm
            # so do not convert back to meters
            pos_out = pos_final_mm

            # transform sensitive axis, not Euler angles
            sens_axis = rot_mat[:, 2]
            sens_axis_final = R_tkras_to_scanner @ (R_mne @ sens_axis)
            sens_axis_final = sens_axis_final / np.linalg.norm(sens_axis_final)

            rows.append([
                name,
                pos_out[0], pos_out[1], pos_out[2],
                sens_axis_final[0], sens_axis_final[1], sens_axis_final[2],
            ])

        pos_file = os.path.join(meg_dir, f"{run_id}_positions.tsv")
        with open(pos_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)

        rows = [["name", "units", "type"]]
        for chan in data.info["chs"]:
            name = f'{chan["ch_name"]}_opm'
            rows.append([name, "nT", "MEG"])

        chan_file = os.path.join(meg_dir, f"{run_id}_channels.tsv")
        with open(chan_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)

        surf_set = LayerSurfaceSet(sub_id, 2)
        config = {
            "data": matlab.double(data.get_data().transpose([1, 2, 0]).tolist()),
            "channels": chan_file,
            "fs": float(data.info["sfreq"]),
            "positions": pos_file,
            "cortex": surf_set.get_mesh_path(),
            "sMRI": surf_set.mri_file,
            "fname": f"spm_{run_id}_epo",
            "path": meg_dir,
        }
        spm.spm_opm_create(config, nargout=0)

        surf_set = LayerSurfaceSet(sub_id, 11)
        config = {
            "data": matlab.double(data.get_data().transpose([1, 2, 0]).tolist()),
            "channels": chan_file,
            "fs": float(data.info["sfreq"]),
            "positions": pos_file,
            "cortex": surf_set.get_mesh_path(),
            "sMRI": surf_set.mri_file,
            "fname": f"spm_{run_id}_11l_epo",
            "path": meg_dir,
        }
        spm.spm_opm_create(config, nargout=0)


if __name__ == "__main__":
    subj_id = sys.argv[1]

    print(f"Processing subject: {subj_id}")
    spm = spm_standalone.initialize()
    process_subject(subj_id, spm)
    spm.terminate()