
from audimg import *
import pickle

ROI = "1014"
subj = "sid001088"
directory = "/isi/music/auditoryimagery2/seanfiles/sid001088/1014/"

ds = get_subject_ds(subj)
rois = [1014]
ds_masked = mask_subject_ds(ds, subj, rois)

pickle.dump(ds_masked, open(directory+"zscoreL_ds.p", "wb"))

ds_masked = mask_subject_ds(ds, subj, rois, True, False)

pickle.dump(ds_masked, open(directory+"nozL_ds.p","wb"))