import glob
import json
import os
import sys
import subprocess

from lameg.surf import LayerSurfaceSet


def run(subj_id):
    os.environ['SUBJECTS_DIR']='/home/common/bonaiuto/laminar_cued_action_opm/derivatives/fs/'
    print('SUBJECT ID: {}'.format(subj_id))

    t1_file = os.path.join('/home/common/bonaiuto/laminar_cued_action_opm/data',subj_id, 'mri', f'{subj_id}_t1.nii')

    cmd=['recon-all','-subjid',subj_id,'-hires','-i',t1_file,'-all','-expert','expert.opts','-parallel','-openmp','8']
    print(' '.join(cmd))
    subprocess.run(cmd)

    surf_set = LayerSurfaceSet(subj_id, 11)
    surf_set.create(
        ds_factor=0.1,
        orientation='link_vector',
        fix_orientation=True
    )

    surf_set = LayerSurfaceSet(subj_id, 2)
    surf_set.create(
        ds_factor=0.1,
        orientation='link_vector',
        fix_orientation=True
    )

if __name__=='__main__':
    try:
        subj_id = sys.argv[1]
    except:
        print("incorrect arguments")
        sys.exit()

    run(subj_id)
