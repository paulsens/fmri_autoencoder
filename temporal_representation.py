import sys
from Constants import *
from temporal_helpers import *
from datetime import datetime
import tensorflow as tf
from random import randint


### THIS FILE SHOULD BE RUN WITH PYTHON3 INCLUDING KERAS, PICKLE, AND NUMPY ###
# with tf.device("/cpu:0"):
args = len(sys.argv)
arg_dict = {}
with tf.device("/cpu:0"):

    if args < 3:
        iteration = sys.argv[1]
        AEs_F = 3
        choice = randint(0, 10)

        arg_dict["subject"] = SUBJECTS[choice]
        arg_dict["ROI"] = "1035"
        arg_dict["task"] = "timbre"
        arg_dict["condition"] = "both"

    else:
        iteration = "NULL"
        AEs_F = int(sys.argv[1])

        arg_dict["subject"] = str(sys.argv[2])

        arg_dict["ROI"] = str(sys.argv[3])

        arg_dict["task"] = str(sys.argv[4])

        arg_dict["condition"] = str(sys.argv[5])

    if args > 5:
        #If we're using user-provided hyperparameters, needed for hyperparameter search
        arg_dict["temporal_window1"] = int(sys.argv[6])
        arg_dict["spatial_window1"] = int(sys.argv[7])
        arg_dict["temporal_window2"] = int(sys.argv[8])
        arg_dict["spatial_window2"] = int(sys.argv[9])
        arg_dict["num_primary_filters"] = int(sys.argv[10])
        arg_dict["num_secondary_filters"] = int(sys.argv[11])
        arg_dict["pooling_cube"] = str(sys.argv[12]).strip().split(",")

    else:
        #Use default values from Constants.py
        temporal_window1_choices = [6, 9]
        spatial_window1_choices = [16, 14]
        temporal_window2_choices = [9, 12]
        spatial_window2_choices = [16, 14]
        prim_filter_choices = [16, 14, 12]
        sec_filter_choices = [4, 5, 6]
        pooling_cube_choices = [[2,2,2], [3,3,3],[3,2,2]]


        choice = randint(0, 1)
        arg_dict["temporal_window1"] = temporal_window1_choices[choice]

        choice = randint(0,1)
        arg_dict["spatial_window1"] = spatial_window1_choices[choice]

        choice = randint(0,1)
        arg_dict["temporal_window2"] = temporal_window2_choices[choice]

        choice = randint(0,1)
        arg_dict["spatial_window2"] = spatial_window2_choices[choice]

        choice = randint(0,2)
        arg_dict["num_primary_filters"] = prim_filter_choices[choice]

        choice = randint(0,2)
        arg_dict["num_secondary_filters"] = sec_filter_choices[choice]

        choice = randint(0,2)
        arg_dict["pooling_cube"] = pooling_cube_choices[choice]




    #log_file = open(PATH+"seanfiles/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/temporal_log_"+str(datetime)+".txt","w")
    log_file = open(PATH+"seanfiles/logs/hpsearch_"+str(iteration)+"_log.txt","w")

    # ADD ANY OTHER HYPERPARAMETERS THAT WE WANT TO TRACK
    log_file.write("-------- HYPERPARAMETERS ---------\n\n")
    for key in arg_dict:
        log_file.write(str(key)+" :    "+str(arg_dict[key])+"\n")
    log_file.write("Z_T_Primary :    "+str(ZERO_THRESHOLD_PRIMARY)+"\n")
    log_file.write("Z_T_Secondary :    "+str(ZERO_THRESHOLD_SECONDARY)+"\n")

    log_file.write("\n\n--------- Training Begins --------")

    #Value 0, 1, 2, or 3
    # 1: Train and save first block
    # 2: Train and save second block, assuming first block already exists
    # 3: Train and save first and second block

    #Data should already be pickled in PATH/seanfiles/subject/ROI/__.p
     # where ___ is either samples, chunks, or targets

    if(AEs_F) in [1, 3]:
        log_file.write("Training primary AE... \n\n")
        ae, window_samples = primary_ae(arg_dict, log_file, "primary")

        #left_3d and right_3d are 3d python lists that are time series of approximate-cubes. They will need to be padded with zeros later to be treated as cubes.
        log_file.write("Performing primary convolution... \n\n")
        left_3d, right_3d = primary_convolution(ae, arg_dict, log_file, "primary", None, None, None)

        log_file.write("Saving first block... \n\n")
        save_block(arg_dict, left_3d, right_3d, "primary", log_file, iteration)

    if(AEs_F) in [2, 3]:
        if(AEs_F == 2):
            left_3d, right_3d = load_block(arg_dict, "primary", 3, log_file, iteration)
            log_file.write("after loading, the length of left_3d is "+str(len(left_3d))+"\n")
        #left and right are python lists with dimensions (num_filters x TIMESTEPS x num_pooled_values)
            #num_pooled_values is not a constant, it is determined in the process by moving the pooling cubes around the space
                #it will be the same value on every timestep of every filter, however.

        #HEMISPHERES_F is defined in Constants.py and controls which half or both halves we use

        secondary_flat_responses, l_count, r_count = flatten_3d_responses(left_3d, right_3d, arg_dict, log_file)
        log_file.write("length of 2nd f r is "+str(len(secondary_flat_responses))+" and its elements have length "+str(len(secondary_flat_responses[0]))+"\n")
        #what are the dimensions we'll need to make a numpy cube padded with zeros?
        #Note that the same cube will work for every time slice, so we can just check the zero'th one.
        spatial_info_l = get_padded_dimensions(left_3d, log_file)
        spatial_info_r = get_padded_dimensions(right_3d, log_file)
        log_file.write("spatial info l is "+str(spatial_info_l)+"\n")

        log_file.write("Training secondary AE... \n\n")
        autoencoders = secondary_ae(arg_dict, secondary_flat_responses, log_file)

        log_file.write("Performing secondary convolution... \n\n")
        secondary_left, secondary_right = secondary_convolution(autoencoders, arg_dict, spatial_info_l, spatial_info_r, secondary_flat_responses, l_count, log_file)

        log_file.write("Saving second block... \n\n")
        save_block(arg_dict, secondary_left, secondary_right, "secondary", log_file, iteration)

    log_file.close()

