import os
import os.path as op
import re
import csv
import subprocess

import numpy as np
import mne
from mne.io.constants import FIFF


BASE_DIR = "/home/bonaiuto/laminar_cued_action_opm/OPM-LMN.260209"

def find_runs(meg_dir):
    files = [f for f in os.listdir(meg_dir) if f.startswith("R") and f.endswith(".fif")]

    def run_key(fname):
        m = re.search(r"R(\d+)", fname)
        return int(m.group(1)) if m else 9999

    files = sorted(files, key=run_key)
    return [op.join(meg_dir, f) for f in files], files


def extract_head_points(raw):
    dig = raw.info["dig"]
    if dig is None:
        raise RuntimeError("raw.info['dig'] is empty")

    fid = {}
    hpi = {}
    hsp = []

    for d in dig:
        r = np.asarray(d["r"], float)

        if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if d["ident"] == FIFF.FIFFV_POINT_LPA:
                fid["LPA"] = r
            elif d["ident"] == FIFF.FIFFV_POINT_NASION:
                fid["NAS"] = r
            elif d["ident"] == FIFF.FIFFV_POINT_RPA:
                fid["RPA"] = r

        elif d["kind"] == FIFF.FIFFV_POINT_HPI:
            hpi[f"coil{int(d['ident'])}"] = r

        elif d["kind"] == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(r)

    return fid, hpi, np.array(hsp, float)


def convert_points_to_mri(raw_fname, trans_fname, subject, subjects_dir):
    raw = mne.io.read_raw_fif(raw_fname, preload=False, verbose=False)
    trans = mne.read_trans(trans_fname)

    fid_head, hpi_head, hsp_head = extract_head_points(raw)

    fid_mri = {
        name: mne.head_to_mri(
            pos[np.newaxis, :],
            subject=subject,
            mri_head_t=trans,
            subjects_dir=subjects_dir,
            kind="mri",
        )[0]
        for name, pos in fid_head.items()
    }

    hpi_mri = {
        name: mne.head_to_mri(
            pos[np.newaxis, :],
            subject=subject,
            mri_head_t=trans,
            subjects_dir=subjects_dir,
            kind="mri",
        )[0]
        for name, pos in hpi_head.items()
    }

    return fid_mri



subjects_dir = "/home/bonaiuto/laminar_cued_action_opm/OPM-LMN.260209/anat/fs/"


def _mat(flag, orig_mgz):
    out = subprocess.check_output(['mri_info', flag, orig_mgz]).decode().split()
    return np.array([float(x) for x in out]).reshape(4, 4)

def tkras_to_scanner_ras(coords_mm):
    # coords_mm: (N,3) tkRAS
    xyz1 = np.c_[coords_mm, np.ones((coords_mm.shape[0], 1))]
    out = (n_orig @ (inv_t_orig @ xyz1.T)).T[:, :3]
    return out


for modality in ['squid','opm']:
    modality_dir = op.join(BASE_DIR, modality)

    OUTPUT_TSV = op.join(modality_dir, "fiducial_coords.tsv")

    subjects = sorted(
        d for d in os.listdir(op.join(modality_dir, 'proc'))
        if d.startswith("S") and op.isdir(op.join(modality_dir, 'proc', d))
    )

    rows = []

    for subject in subjects:

        meg_dir = op.join(modality_dir, 'proc', subject)


        orig_mgz = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')

        t_orig = _mat('--vox2ras-tkr', orig_mgz)  # vox -> tkRAS (mm)
        n_orig = _mat('--vox2ras', orig_mgz)  # vox -> scanner RAS (mm)
        inv_t_orig = np.linalg.inv(t_orig)

        run_paths, run_names = find_runs(meg_dir)

        for fname, name in zip(run_paths, run_names):

            run_id = name.split(".")[0]

            if modality=='squid':
                raw_fname=op.join(modality_dir, 'raw', subject, f'{run_id}.tsss.fif')
            else:
                raw_fname = op.join(modality_dir, 'raw', subject, f'{run_id}.fif')
            print(f"\nReading {raw_fname}")

            trans_fname = op.join(modality_dir, "trans", subject, f"{run_id}-trans.fif")

            fid_mri = convert_points_to_mri(raw_fname, trans_fname, subject, subjects_dir)
            for fid in fid_mri:
                fid_mri[fid] = np.squeeze(tkras_to_scanner_ras(fid_mri[fid][np.newaxis, :]))

            nas = ",".join([f"{x:.6f}" for x in fid_mri["NAS"]])
            lpa = ",".join([f"{x:.6f}" for x in fid_mri["LPA"]])
            rpa = ",".join([f"{x:.6f}" for x in fid_mri["RPA"]])

            rows.append([
                f"{subject}-{run_id}",
                nas,
                lpa,
                rpa
            ])


    with open(OUTPUT_TSV, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["subject_id", "NAS", "LPA", "RPA"])
        writer.writerows(rows)

    print(f"\nSaved TSV: {OUTPUT_TSV}")