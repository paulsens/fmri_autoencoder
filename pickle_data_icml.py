from Constants import *
import mvpa2.suite as P

import bids
import bids.grabbids as gb
from os.path import join as opj
import csv
import glob
import pickle
import os
import numpy as np

ROOTDIR='/isi/music/auditoryimagery2'
DATADIR=opj(ROOTDIR, 'am2/data/fmriprep/fmriprep/')
OUTDIR=opj(ROOTDIR, 'results_audimg_subj_task')

MRISPACE= 'MNI152NLin2009cAsym' # if using fmriprep_2mm then MRISPACE='MNI152NLin6Asym'
PARCELLATION='desc-aparcaseg_dseg'# per-subject ROI parcellation in MRIa

# List of tasks to evaluate

def _make_subj_id_maps():
    """
    Utility function
    Read subj-id-accession-key.csv to map ids to accession number and tonality (E or F)
    """
    global subjects, accessions, tonalities, subjnums
    subjects = []
    accessions = {}
    tonalities = {}
    subjnums = {}
    with open(opj(ROOTDIR,'subj-id-accession-key.csv')) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row = reader.next() # header row
        for row in reader:
            subj_num, subj_id, accession_num, key = row
            subj_id = subj_id.lower()
            #print subj_num, subj_id, accession_num, key
            subjects.append(subj_id)
            subjnums[subj_num] = subj_id
            accessions[subj_id] = accession_num
            tonalities[subj_id] = key

def _make_legend(LEGEND_FILE=opj(ROOTDIR,"subj_task_run_list.txt")):
    """
    Utility function
    Read run_legend to re-order runs by task
    We could hard-code this as a dict, only need the first row for run-order
    """
    global legend
    legend = {}
    with open(LEGEND_FILE,"rt") as f:
        try:
            lines = [line.strip() for line in f.readlines()]
        except:
            raise IOError("Cannot open %s for reading"%LEGEND_FILE)
        for line in lines:
            line = line.split('-') # sid001125_task - pitchheardXtrumXF_run - 01
            subj_id = line[0].split('_')[0]
            run = int(line[-1])
            task=line[1][5].upper() # H or I
            task+=line[1].split('X')[1][0].upper() # T or C
            if run==1:
                legend[accessions[subj_id]]=[]
            legend[accessions[subj_id]].append(task)


def _gen_RH_cortical_map():
    """
    Utility function to generate RH of cortical map
    """
    roi_map.pop(1004) # The Corpus Callosum is not defined
    for k in roi_map.keys():
        roi_map[k+1000]=roi_map[k].replace('lh','rh')

def get_subject_ds(subject, cache=False, cache_dir='ds_cache'):
    """Assemble pre-processed datasets
    load subject original data (no mask applied)
    optionally cache for faster loading during model training/testing
    purpose: re-use unmasked dataset, applying mask when necessary

    inputs:
        subject  - sid00[0-9]{4}
        cache    - whether to use cached datasets [False]
     cache_dir   - where to store / load cached datasets ['ds_cache']

    outputs:
        data     - subject original data (no mask applied)
    """
    swap_timbres=[2,1,4,3,6,5,8,7]
    layout = gb.BIDSLayout(DATADIR)
    ext = 'desc-preproc_bold.nii.gz'
    cache_filename = '%s/%s.ds_cache.nii.gz'%(cache_dir, subject)
    cache_lockname = '%s/%s.ds_cache.lock'%(cache_dir, subject)
    cache_fail=False
    if cache:
        try:
            data=P.Dataset.from_hdf5(cache_filename)
        except:
            cache_fail=True
    if not cache or cache_fail:
        data=[]
        for run in range(1,9):
#            r=run if legend[accessions[subject]][0]=='HT' else swap_timbres[run-1]
            r=run
            f=layout.get(subject=subject, extensions=[ext], run=r)[0]
            tgts=np.loadtxt(opj(ROOTDIR + '/targets', ACCESSIONS[subject]+'_run-%02d.txt'%r)).astype('int')
            ds = P.fmri_dataset(f.filename,
                             targets=tgts,
                             chunks=run)
            if not ds.shape[1]:
                raise ValueError("Got zero mask (no samples)")
            #print "subject", subject, "chunk", run, "run", r, "ds", ds.shape
            data.append(ds)
        data=P.vstack(data, a=0)
        if cache and not os.path.exists(cache_lockname):
            with open(cache_lockname, "w") as f:
                f.write("audimg lock\n");
                f.close()
                P.Dataset.save(data, cache_filename, compression='gzip')
                os.remove(cache_lockname)
    return data


def get_subject_mask(subject, run=1, rois=[1030, 2030], path=DATADIR,
                     space=MRISPACE,
                     parcellation=PARCELLATION):
    """
    Get subject mask by run and ROI key to apply to a dataset
    (rois are in DATADIR/PARCELLATION.tsv)

    inputs:
        subject  - sid00[0-9]{4}
        run      - which run to use for parcellation (redundant?) [1-8]
        rois     - list of regions of interest for mask [1030,2030]
        path     - dir containing roi parcellations [DATADIR]
       space     - parcellation space [MRISPACE]
     parcellation- file  [PARCELLATION]

    outputs:
        mask_ds  - pymvpa Dataset containing mask data {0,[rois]}
    """
    fname = opj(path, 'sub-%s' % subject, 'func',
                'sub-%s_task-*_run-%02d_space-%s_%s.nii.gz' % (subject, run, space, parcellation))
    # print fname
    fname = glob.glob(fname)[0]
    ds = P.fmri_dataset(fname)
    found = np.where(np.isin(ds.samples, rois))[1]
    return ds[:, found]


def mask_subject_ds(ds, subj, rois, detrend=True, zscore=True):
    """
    Mask a subject's data for given list of rois

    inputs:
         ds - the dataset to mask
       subj - sid00[0-9]{4}
       rois - list of rois to merge e.g. [1005, 1035, 2005, 2035]
    detrend - remove trend from roi dataset [True]
     zscore - voxel-wise z-scoring of roi dataset [True]

    outputs:
     ds_masked - the masked dataset (data is copied)
    """
    if subj is not None:
        mask = get_subject_mask('%s' % subj, run=1, rois=rois)
        ds_masked = P.fmri_dataset(P.map2nifti(ds), ds.targets, ds.chunks, P.map2nifti(mask))
        if detrend:
            P.poly_detrend(ds_masked, polyord=1, chunks_attr='chunks')  # in-place
        if zscore:
            P.zscore(ds_masked, param_est=('targets', [1, 2]))  # in-place
    else:
        ds_masked = ds.copy()
    return ds_masked
#
roi_map={
1000:    "ctx-lh-unknown",        #190519
1001:    "ctx-lh-bankssts",       #196428
1002:    "ctx-lh-caudalanteriorcingulate", #7d64a0
1003:    "ctx-lh-caudalmiddlefrontal",    #641900
#1004:    "ctx-lh-corpuscallosum", #784632
1005:    "ctx-lh-cuneus", #dc1464
1006:    "ctx-lh-entorhinal",     #dc140a
1007:   "ctx-lh-fusiform",       #b4dc8c
1008:    "ctx-lh-inferiorparietal",       #dc3cdc
1009:    "ctx-lh-inferiortemporal",       #b42878
1010:    "ctx-lh-isthmuscingulate",       #8c148c
1011:    "ctx-lh-lateraloccipital",       #141e8c
1012:    "ctx-lh-lateralorbitofrontal",   #234b32
1013:    "ctx-lh-lingual",        #e18c8c
1014:    "ctx-lh-medialorbitofrontal",    #c8234b
1015:    "ctx-lh-middletemporal", #a06432
1016:    "ctx-lh-parahippocampal",        #14dc3c
1017:    "ctx-lh-paracentral",    #3cdc3c
1018:    "ctx-lh-parsopercularis",        #dcb48c
1019:    "ctx-lh-parsorbitalis",  #146432
1020:    "ctx-lh-parstriangularis",       #dc3c14
1021:    "ctx-lh-pericalcarine",  #78643c
1022:    "ctx-lh-postcentral",    #dc1414
1023:    "ctx-lh-posteriorcingulate",     #dcb4dc
1024:    "ctx-lh-precentral",     #3c14dc
1025:    "ctx-lh-precuneus",      #a08cb4
1026:    "ctx-lh-rostralanteriorcingulate",       #50148c
1027:    "ctx-lh-rostralmiddlefrontal",   #4b327d
1028:    "ctx-lh-superiorfrontal",        #14dca0
1029:    "ctx-lh-superiorparietal",       #14b48c
1030:    "ctx-lh-superiortemporal",       #8cdcdc
1031:    "ctx-lh-supramarginal",  #50a014
1032:    "ctx-lh-frontalpole",    #640064
1033:    "ctx-lh-temporalpole",   #464646
1034:    "ctx-lh-transversetemporal",     #9696c8
1035:    "ctx-lh-insula" #ffc020
}

#_make_subj_id_maps()

#_make_legend()

#_gen_RH_cortical_map()

tasks=['pch-height','pch-class','pch-hilo','timbre','pch-helix-stim-enc']
ROI_dict = {tasks[0]:1032, tasks[1]:1001, tasks[2]:1034, tasks[3]:1013, tasks[4]:1034}

icmlsubjects= ['sid001088']
def pickle_data_new():
    subject_count = 0
    ds_count = 0

    ### SUBJECTS is set in Constants.py ###
    for subject in icmlsubjects:
        if str(subject) in RED_FLAGGED:
            print("Subject "+str(subject)+" is red-flagged, ignoring.\n\n")

        else:

            subject_count += 1

            if str(subject) in YELLOW_FLAGGED:
                print("Subject "+str(subject)+" is yellow-flagged, allowing.\n\n")

            ds = get_subject_ds(subject)
            print("about to print ds object")
            print(ds)



            pickle.dump(ds, open("/isi/music/auditoryimagery2/seanfiles/icml/sid001088/full_brain_ds.p", "wb"))
            ds_count += 1
            #
            # pickle.dump(samples, open(directory+"/samples.p", "wb"))
            # pickle.dump(orig_data,open(directory+"/3d_samples.p","wb"))
            # pickle.dump(voxels, open(directory+"/voxel_indices.p","wb"))
            # pickle.dump(chunks, open(directory+"/chunks.p", "wb"))
            # pickle.dump(targets, open(directory+"/targets.p", "wb"))

    print("Loaded " + str(subject_count) + " subjects.\n\n")

