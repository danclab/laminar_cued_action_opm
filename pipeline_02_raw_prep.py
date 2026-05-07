import copy
import os
import os.path as op
import re
import sys

from meegkit.dss import dss_line
import mne
from mne.time_frequency import psd_array_multitaper

import numpy as np
import matplotlib

from utilities import files

matplotlib.use('Agg')
import matplotlib.pylab as plt



BASE_DIR = "/home/common/bonaiuto/laminar_cued_action_opm/derivatives"
sfreq=250

def find_runs(meg_dir, modality):
    if modality=='squid':
        files = [f'R{r_idx+1}.tsss.fif' for r_idx in range(5)]
    else:
        files = [f'R{r_idx + 1}.fif' for r_idx in range(5)]
    return [op.join(meg_dir, f) for f in files], files

def nan_basic_interp(array):
    nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
    array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
    return array


def zapline_until_gone(data, target_freq, sfreq, win_sz=10, spot_sz=2.5, viz=False, prefix="zapline_iter",
                       max_iter=100):
    """
    Returns: clean data, number of iterations

    Function iteratively applies the Zapline algorithm.

    data: assumed that the function is a part of the MNE-Python pipeline,
    the input should be an output of {MNE object}.get_data() function. The shape
    should be Trials x Sensors x Time for epochs.
    target_freq: frequency + harmonics that comb-like approach will be applied
    with Zapline
    sfreq: sampling frequency, the output of {MNE object}.info["sfreq"]
    win_sz: 2x win_sz = window around the target frequency
    spot_sz: 2x spot_sz = width of the frequency peak to remove
    viz: produce a visual output of each iteration,
    prefix: provide a path and first part of the file
    "{prefix}_{iteration number}.png"
    """

    iterations = 0
    aggr_resid = []

    freq_rn = [target_freq - win_sz, target_freq + win_sz]
    freq_sp = [target_freq - spot_sz, target_freq + spot_sz]

    norm_vals = []
    resid_lims = []
    real_target=None

    while True:
        if iterations > 0:
            if iterations >= max_iter:
                break
            if real_target is None:
                real_target=freq[freq_rn_ix[0]:freq_rn_ix[1]][np.argmax(mean_psd)]
                print(f'Real target={real_target}')
            data, art = dss_line(data.transpose(), real_target, sfreq, nremove=1)
            del art
            data = data.transpose()
        psd, freq = psd_array_multitaper(data, sfreq, verbose=False, n_jobs=30)

        freq_rn_ix = [
            np.where(freq >= freq_rn[0])[0][0],
            np.where(freq <= freq_rn[1])[0][-1]
        ]
        freq_used = freq[freq_rn_ix[0]:freq_rn_ix[1]]
        freq_sp_ix = [
            np.where(freq_used >= freq_sp[0])[0][0],
            np.where(freq_used <= freq_sp[1])[0][-1]
        ]

        norm_psd = psd[:, freq_rn_ix[0]:freq_rn_ix[1]]
        for ch_idx in range(norm_psd.shape[0]):
            if iterations == 0:
                norm_val = np.max(norm_psd[ch_idx, :])
                norm_vals.append(norm_val)
            else:
                norm_val = norm_vals[ch_idx]
            norm_psd[ch_idx, :] = norm_psd[ch_idx, :] / norm_val
        mean_psd = np.mean(norm_psd, axis=0)

        mean_psd_wospot = copy.copy(mean_psd)
        mean_psd_wospot[freq_sp_ix[0]: freq_sp_ix[1]] = np.nan
        mean_psd_tf = nan_basic_interp(mean_psd_wospot)
        pf = np.polyfit(freq_used, mean_psd_tf, 3)
        p = np.poly1d(pf)
        clean_fit_line = p(freq_used)
        residuals = mean_psd - clean_fit_line
        aggr_resid.append(np.mean(residuals))
        tf_ix = np.where(freq_used <= target_freq)[0][-1]
        print("Iteration:", iterations, "Power above the fit:", residuals[tf_ix])

        if viz:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6), facecolor="gray", gridspec_kw={"wspace": 0.2})
            for sensor in range(psd.shape[0]):
                ax1.plot(freq_used, norm_psd[sensor, :])
            ax1.set_title("Normalized mean PSD \nacross trials")

            ax2.plot(freq_used, mean_psd_tf, c="gray")
            ax2.plot(freq_used, mean_psd, c="blue")
            ax2.plot(freq_used, clean_fit_line, c="red")
            ax2.set_title("Mean PSD across \ntrials and sensors")

            ax3.set_title("Residuals")
            tf_ix = np.where(freq_used <= target_freq)[0][-1]
            ax3.plot(residuals, freq_used)
            scat_color = "green"
            if residuals[tf_ix] <= 0:
                scat_color = "red"
            ax3.scatter(residuals[tf_ix], freq_used[tf_ix], c=scat_color)
            if iterations == 0:
                resid_lims = ax3.get_xlim()
            else:
                ax3.set_xlim(resid_lims)

            ax4.set_title("Iterations")

            ax4.scatter(np.arange(iterations + 1), aggr_resid)
            plt.savefig("{}_{}.png".format(prefix, str(iterations).zfill(3)))
            plt.close("all")

        if iterations > 0 and residuals[tf_ix] <= 0:
            break

        iterations += 1

    return [data, iterations]


def process_subject(sub_dir, modality):
    meg_dir = op.join(sub_dir, modality, "meg")
    behav_dir = op.join(sub_dir, modality, "behav")
    run_paths, run_names = find_runs(meg_dir, modality)
    qc_folder = op.join(meg_dir, "QC")
    files.make_folder(qc_folder)

    for path, name in zip(run_paths, run_names):
        print(f"\nReading {path}")
        run_id = name.split('.')[0]

        raw_sss = mne.io.read_raw_fif(path, preload=True, verbose=False)

        raw_events = mne.find_events(
            raw_sss,
            stim_channel="STI101",
            shortest_event=1,
            verbose="DEBUG",
            consecutive=True
        )
        raw_events[:, 2] = 1
        if modality=='squid':
            raw_events[:, 0] += int(-0.010 * 1000)
        else:
            raw_events[:, 0] += int(0.045 * 1000)

        behv = np.loadtxt(op.join(behav_dir, f"{run_id}.txt"), delimiter=",")
        good_idx = np.where((behv[:, 1] == 0) & (behv[:, 4] == 0))[0]
        raw_events = raw_events[good_idx,:]

        eve_path = op.join(
            meg_dir,
            "{}-eve.fif".format(run_id)
        )
        raw_sss.notch_filter(freqs=[50, 100, 150, 200, 250], method="fir", n_jobs=-1, verbose=False)

        raw_sss.filter(l_freq=1, h_freq=250, method="fir", n_jobs=-1, verbose=False)

        raw_sss, events = raw_sss.copy().resample(
            sfreq,
            npad="auto",
            events=raw_events,
            n_jobs=-1,
        )


        fig = raw_sss.plot_psd(
            tmax=np.inf, fmax=125, average=True, show=False, picks="meg"
        )
        fig.suptitle(run_id)
        plt.savefig(
            op.join(qc_folder, "{}-raw-psd.png".format(run_id)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        info = raw_sss.info
        meg_picks = mne.pick_types(info, meg=True, ref_meg=False)
        raw_meg = raw_sss.get_data(picks=meg_picks)

        zapped, iterations = zapline_until_gone(
            raw_meg,
            50.0,
            info['sfreq'],
            win_sz=5,
            spot_sz=2,
            viz=True,
            prefix="{}/{}-50_1_iter".format(qc_folder, run_id),
            max_iter=0
        )

        full_data = raw_sss.get_data()
        full_data[meg_picks, :] = zapped

        raw_sss = mne.io.RawArray(full_data, info)
        fig = raw_sss.plot_psd(tmax=np.inf, fmax=125, average=True, show=False, picks='meg')
        fig.suptitle(run_id)
        plt.savefig(
            op.join(qc_folder, "{}-zapline-raw-psd.png".format(run_id)),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close("all")

        out_path = op.join(
            meg_dir,
            "zapline-{}-raw.fif".format(run_id)
        )

        raw_sss.save(
            out_path,
            overwrite=True
        )
        mne.write_events(
            eve_path,
            events,
            overwrite=True
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

