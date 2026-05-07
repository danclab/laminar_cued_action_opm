import sys
import json
import mne
import os
import os.path as op
import subprocess as sp
import numpy as np
from utilities import files
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    run_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

modality = sys.argv[3]

BASE_DIR = "/home/common/bonaiuto/laminar_cued_action_opm/derivatives"

subjects = files.get_folders_files(BASE_DIR)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]
print("ID:", subject_id)

sub_path = op.join(BASE_DIR, subject_id)

meg_path = op.join(sub_path, modality, "meg")

qc_folder = op.join(meg_path, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(meg_path, "zapline-", "-raw.fif")[2]
raw_paths.sort()
raw_path = raw_paths[run_index]

event_paths = files.get_files(meg_path, 'R', "-eve.fif")[2]
event_paths.sort()
event_path = event_paths[run_index]

ica_paths = files.get_files(meg_path, "R", "-ica.fif")[2]
ica_paths.sort()
ica_path = ica_paths[run_index]

ica_json_file = op.join(
    meg_path,
    "ICA_to_reject.json"
)


print("SUBJ: {}".format(subject_id), run_index)
print("INPUT RAW FILE:", raw_path.split(os.sep)[-1])
print("INPUT EVENT FILE:", event_path.split(os.sep)[-1])
print("INPUT ICA FILE:", ica_path.split(os.sep)[-1])
print("INPUT JSON FILE", ica_json_file.split(os.sep)[-1])

raw = mne.io.read_raw_fif(
    raw_path, preload=True, verbose=False
)

events = mne.read_events(event_path)

ica = mne.preprocessing.read_ica(
    ica_path, verbose=False
)

raw.crop(
    tmin=raw.times[events[0,0]]
)
raw.filter(1,20, verbose=False)
raw.close()

sp.Popen(
    ["mousepad", str(ica_json_file)],
    stdout=sp.DEVNULL,
    stderr=sp.DEVNULL
)
print('')

title_ = "sub:{}, file: {}".format(subject_id, ica_path.split(os.sep)[-1])

ica.plot_components(inst=raw, picks=np.arange(25), show=False, title=title_)

ica.plot_sources(inst=raw, picks=np.arange(25), show=False, title=title_)

plt.show()

