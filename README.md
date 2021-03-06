# fmri_autoencoder
Pipeline for transforming fMRI data as detailed in May, Halpern, Paulsen, and Casey(2022)

**Step 0**: Python environment
  
- **0.1**: All code in this repository should be run with a Python2 environment which has the pymvpa2, pybids (version <= 0.6.5), numpy, scikit-learn, pickle, tensorflow (v. 1.13.1) and keras (v. 2.3.1). We encountered issues with bids and/or pymvpa2 in Python3. As of March 28 2022, I was able to obtain all the necessary packages in a fresh Python 2.7 conda environment. 

**Step 1**: ROI Slicing
  
- **1.1**: Email Michael.A.Casey@dartmouth.edu to request access to our brainlife.io repository of fMRI data and then download it.
  
- **1.2**: Download the "Extras" folder located at https://drive.google.com/file/d/1lXT0wM-Nw6gKdaCWJjG9dSgGFaVKGzLA/view?usp=sharing and extract the contents.
  
- **1.3**: Set the ROI_PATH, CODE_PATH, DATA_PATH, EXTRAS_PATH, and LOGS_PATH variables in Constants.py
  
- **1.4**: Activate your python environment and run prepare_data.py. This will extract the data from .nii files and create corresponding mvpa2 "dataset" objects for each of the ten (technically twenty, as left and right are handled separately) Regions of Interest. These objects are then saved on disk with pickle and are used throughout the rest of the pipeline. 
  
**Step 2**: Transformed Datasets
  
- **2.1**: Run make_ds.py with three command line arguments. First, the full subject id, e.g "sub-sid00XXXX". Second, the left-hand ROI, e.g "10XX". The pipeline will run for both left and right side with only the left hand given. Third, either a 0 or 1, e.g "python make_ds.py 0". The 0 or 1 indicates the choice of strategy for training the autoencoder. If 0, the autoencoder for each subject will be trained on all of that subject's UNLABELLED data, and no task data. This was our original approach but raised concerns of data circularity. If 1, two autoencoders will be trained for each subject. The first is trained on the first four of eight runs, and will ultimately be used to "transform" runs 5 through 8. The second is trained on runs 5 through 8 and is used to "transform" runs 1 through 4. This was our approach to avoid data circularity. In summary, a command line execution to create, for example, the "transformed" dataset objects for regions 1030 and 2030 of subject 1234 that avoids data circularity would be "python make_ds.py sub-sid001234 1030 1"
- **2.2** In the case of strategy 1, make_ds.py ends with the two "transformed" halves saved separately. For each subject, run combine_halves.py with the full subject id as an argument, e.g "python combine_halves.py sid-sub001234".
- **2.3** The final "transformed" datasets will be in ROI_PATH/sid-sub00XXXX/10XX/ and named "strat0_transformed_lh.p", "strat0_transformed_rh.p", "strat1_transformed_lh.p", and "strat1_transformed_rh.p"
**Step 3**: Classifiers

- **3.1**: Follow the instructions at https://github.com/soundspotter/audim_mvpa
        
