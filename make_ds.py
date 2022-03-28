import sys
from Constants import *
from temporal_helpers_polish import *
from keras.models import model_from_json
from datetime import datetime
import tensorflow as tf
from random import randint
import numpy as np
from mvpa2.datasets import *
from mvpa2.base.dataset import save

### THIS FILE SHOULD BE RUN WITH PYTHON3 INCLUDING KERAS, PICKLE, AND NUMPY ###
# with tf.device("/cpu:0"):
args = len(sys.argv)
arg_dict = {}
PATH = "/isi/music/auditoryimagery2/"

tasks = ['pch-height', 'pch-class', 'pch-hilo', 'timbre', 'pch-helix-stim-enc']
task = 'pch-height'
ROI_dict = {tasks[0]: 1032, tasks[1]: 1001, tasks[2]: 1034, tasks[3]: 1013, tasks[4]: 1034}
ROI = ROI_dict[task]
#rois = [ROI, ROI + 1000]  # left and right sides of ROI
rois = [ROI]
ROI = str(ROI_dict[task])
global TIMESTEPS

with tf.device("/cpu:0"):
    if True:
        # iteration = sys.argv[1]

        AEs_F = 3
        choice = randint(0, 10)

        #arg_dict["subject"] = 'sid001401'
        arg_dict["subject"] = str(sys.argv[1])
        arg_dict["ROI"] = str(sys.argv[2])
        strategy = str(sys.argv[3])
        half = int(sys.argv[4])
        arg_dict["strategy"]=strategy
        arg_dict["half"] = half
        if(half!=0):
            TIMESTEPS=932
            arg_dict["TIMESTEPS"] = TIMESTEPS
        else:
            arg_dict["TIMESTEPS"] = 1864
        arg_dict["start_step"] = (half-1)*TIMESTEPS
        hpconfig = 3




        #arg_dict["ROI"] = "1035"
        iteration = arg_dict["subject"] + "_" + arg_dict["ROI"] + "_strat0_half" + str(half)
        #directory = PATH+"seanfiles/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/"


        #these two are irrelevant right now
        arg_dict["task"] = None
        arg_dict["condition"] = "both"

    else:
        iteration = "NULL"
        AEs_F = int(sys.argv[1])

        arg_dict["subject"] = str(sys.argv[2])

        arg_dict["ROI"] = str(sys.argv[3])

        arg_dict["task"] = str(sys.argv[4])

        arg_dict["condition"] = str(sys.argv[5])

    if False:
        # If we're using user-provided hyperparameters, needed for hyperparameter search
        arg_dict["temporal_window1"] = int(sys.argv[6])
        arg_dict["spatial_window1"] = int(sys.argv[7])
        arg_dict["temporal_window2"] = int(sys.argv[8])
        arg_dict["spatial_window2"] = int(sys.argv[9])
        arg_dict["num_primary_filters"] = int(sys.argv[10])
        arg_dict["num_secondary_filters"] = int(sys.argv[11])
        arg_dict["pooling_cube"] = str(sys.argv[12]).strip().split(",")

    else:
        # Use default values from Constants.py
        temporal_window1_choices = [6, 9]
        spatial_window1_choices = [16, 16]
        temporal_window2_choices = [9, 12]
        spatial_window2_choices = [16, 14]
        prim_filter_choices = [8, 16]
        sec_filter_choices = [4, 5, 6]
        pooling_cube_choices = [[1, 1, 2], [3, 2, 2]]

        hps = [ [6, 10, [2, 2, 2]],
                [6, 10, [3, 2, 2]],
                [6, 14, [2, 2, 2]],
                [6, 14, [3, 2, 2]],
                [9, 10, [2, 2, 2]],
                [9, 10, [3, 2, 2]],
                [9, 14, [2, 2, 2]],
                [9, 14, [3, 2, 2]]
             ]
        #which hyperparameter configuration are we using? supplied via commad line
        hpconfig = hpconfig-1
        #choice = hpconfig-1
        # choice = randint(0, 1)
        arg_dict["temporal_window1"] = hps[hpconfig][0]

        # choice = randint(0,1)
        arg_dict["spatial_window1"] = spatial_window1_choices[0]

        # choice = randint(0,1)
        arg_dict["temporal_window2"] = temporal_window2_choices[0]

        # choice = randint(0,1)
        arg_dict["spatial_window2"] = spatial_window2_choices[0]

        # choice = randint(0,2)
        arg_dict["num_primary_filters"] = hps[hpconfig][1]

        # choice = randint(0,2)
        arg_dict["num_secondary_filters"] = sec_filter_choices[0]

        # choice = randint(0,2)
        arg_dict["pooling_cube"] = hps[hpconfig][2]

    # log_file = open(PATH+"seanfiles/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/temporal_log_"+str(datetime)+".txt","w")
    #log_file = open(PATH + "seanfiles/strat"+str(strategy)+"/logs/hpsearch_" +str(iteration) + ".txt", "w")
    log_file = open(PATH+"seanfiles/translurm_logs/"+str(iteration)+".txt","w")

    confirm_file = open("/isi/music/auditoryimagery2/seanfiles/translurm_confirmations/"+str(iteration)+".txt","w")
    confirm_file.write("confirmed")
    confirm_file.close()


    # ADD ANY OTHER HYPERPARAMETERS THAT WE WANT TO TRACK
    log_file.write("-------- HYPERPARAMETERS ---------\n\n")
    for key in arg_dict:
        log_file.write(str(key) + " :    " + str(arg_dict[key]) + "\n")
    log_file.write("Z_T_Primary :    " + str(ZERO_THRESHOLD_PRIMARY) + "\n")
    log_file.write("Z_T_Secondary :    " + str(ZERO_THRESHOLD_SECONDARY) + "\n")

    log_file.write("\n\n--------- Training Begins --------")

    # Value 0, 1, 2, or 3
    # 1: Train and save first block
    # 2: Train and save second block, assuming first block already exists
    # 3: Train and save first and second block

    # Data should already be pickled in PATH/seanfiles/subject/ROI/__.p
    # where ___ is either samples, chunks, or targets
    #

    strat_samples, new_steps, strat_targets, strat_steps, labelled_steps = make_StratDS(arg_dict, log_file)
    arg_dict["strat_samples"]=strat_samples
    arg_dict["strat_steps"]=strat_steps #flat number of timesteps

    arg_dict["new_steps"]=new_steps #list of timesteps
    arg_dict["strat_targets"]=strat_targets
    arg_dict["labelled_steps"]=labelled_steps


    if True:
        log_file.write("Training primary AE... \n\n")
        best_loss = -1
        best_ae = None
        ae_list = []
        for j in range(0,12):
            ae, window_samples = primary_ae(arg_dict, log_file, "primary")
            if strategy==0:
                TIMESTEPS=932
            else:
                TIMESTEPS=arg_dict["strat_steps"]
            ae_list.append(ae)
            log_file.write("AE number "+str(j)+" ended with a training loss of "+str(current_loss[0])+"\n")
            #current_loss is defined and updated within temporal_helpers.py
            #it's a global variable and is used in the LossCallback function.
            if (best_loss == -1) or (current_loss[0] < best_loss):
                best_loss = current_loss[0]
                best_ae = ae

    #    log_file.write("The AEs as strings: "+str(ae_list[0])+" , "+str(ae_list[1])+" , "+str(ae_list[2])+"\n")
     #   log_file.write("The best AE is "+str(best_ae)+"\n\n")




        print("converting to json..\n")
        model_json = best_ae.to_json()

        with open(PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/"+"half"+str(half)+"_primary_ae.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        best_ae.save_weights(PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/half"+str(half)+"_primary_ae.h5")
        print("Saved model to disk")

        #left_3d and right_3d are 3d python lists that are time series of approximate-cubes. They will need to be padded with zeros later to be treated as cubes.

    #load json and create model
    # jsonpath = PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/half"+str(half)+"_primary_ae.json"
    # print("jsonpath is "+str(jsonpath))
    # json_file = open(jsonpath, 'r')
    # loaded_model_json = json_file.read()
    # loaded_model = model_from_json(loaded_model_json)
    #
    # # load weights into new model
    # loaded_model.load_weights(PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/half"+str(half)+"_primary_ae.h5")
    # print("Loaded model from disk")
    # if(half<1):
    #     TIMESTEPS = 1864
    # else:
    #     TIMESTEPS = 932

################ bilateral convolution
    # print("Performing primary convolution for both side... \n\n")
    # arg_dict["ROI"] = arg_dict["ROI"]+"B"
    # print("argidct ROi is "+str(arg_dict["ROI"])+"\n")
    # flat_pooled_B, bilateral_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None, None)
    #
    # print("flat pooled b has length "+str(len(flat_pooled_B))+"\n")
    # print("its second axis length is "+str(len(flat_pooled_B[0]))+"\n")
    # print("bilateral count is "+str(bilateral_count)+"\n")
    # log_file.write("Saving first block... \n\n")
    # arg_dict["ROI"] = arg_dict["ROI"]+"B"
    # save_block(arg_dict, flat_pooled_B, None, "primary", log_file, iteration)
    # arg_dict["ROI"] = arg_dict["ROI"][:-1]
    #
    # log_file.write("first block bilateral saved")
    #
    # filters = len(flat_pooled_B)
    # bilateral_features = len(flat_pooled_B[0][0])
    # print("filters is "+str(filters))
    # print("bilateral features is "+str(bilateral_features))
    # log_file.write("bilateral features is "+str(bilateral_features)+"\n")
    #
    # bilateralarray = np.zeros((1864, bilateral_features*filters))
    # log_file.write("bilateralarray shape is "+str(bilateralarray.shape))
    # print("bilateral array shape is "+str(bilateralarray.shape))
    # # for t in range(0, 1864):
    # #     if(len(flat_pooled_B[3][t])!=162):
    # #         print("ERROR! for t="+str(t)+" the length is "+str(len(flat_pooled_B[3][t]))+"\n")
    # #         print("and the rest of its timesteps are: \n")
    # #         for s in range(t, 1864):
    # #             print(str(s)+":  "+str(len(flat_pooled_B[3][s]))+"\n")
    # #         print("and for filter 4 it's "+str(len(flat_pooled_B[4][t]))+"\n")
    # for t in range(0, 1864):
    #     #if(True):
    #      #   print("t is "+str(t)+"\n\n")
    #     for filter in range(0, filters):
    #        # if(True):
    #         #    print("filter is "+str(filter)+"\n")
    #         for feature in range(0, bilateral_features):
    #           #  if(True):
    #            #     print("feature is "+str(feature)+"\n")
    #             bilateralarray[t][bilateral_features*filter + feature] = flat_pooled_B[filter][t][feature]

################ left side convolution
#     log_file.write("Performing primary convolution for left side... \n\n")
#     arg_dict["ROI"] = arg_dict["ROI"]+"L"
#     flat_pooled_L, left_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None, None)
#
#
#     log_file.write("Saving first block... \n\n")
#     arg_dict["ROI"] = arg_dict["ROI"]+"L"
#
#     save_block(arg_dict, flat_pooled_L, None, "primary", log_file, iteration)
#     arg_dict["ROI"] = arg_dict["ROI"][:-1]
#
#     log_file.write("first block left side saved")
#
#     filters = len(flat_pooled_L)
#     left_features = len(flat_pooled_L[0][0])
#
#     log_file.write("left features is "+str(left_features)+"\n")
#
#     leftarray = np.zeros((TIMESTEPS, left_features*filters))
#     log_file.write("leftarray shape is "+str(leftarray.shape))
#
#     for t in range(0, TIMESTEPS):
#         for filter in range(0, filters):
#             for feature in range(0, left_features):
#                 leftarray[t][left_features*filter + feature] = flat_pooled_L[filter][t][feature]
#
# ################## right side convolution
#     log_file.write("Performing primary convolution for right side, the arg_dict value is "+str(arg_dict["ROI"])+"... \n\n")
#     arg_dict["ROI"] = arg_dict["ROI"]+"R"
#     flat_pooled_R, right_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None, None)
#
#
#     log_file.write("Saving first block... \n\n")
#     arg_dict["ROI"] = arg_dict["ROI"]+"R"
#
#     save_block(arg_dict, flat_pooled_R, None, "primary", log_file, iteration)
#     arg_dict["ROI"] = arg_dict["ROI"][:-1]
#
#     log_file.write("first block left side saved")
#
#     filters = len(flat_pooled_R)
#     right_features = len(flat_pooled_R[0][0])
#
#     log_file.write("right features is "+str(right_features)+"\n")
#
#     rightarray = np.zeros((TIMESTEPS, right_features*filters))
#     log_file.write("rightarray shape is "+str(rightarray.shape))
#
#     for t in range(0, TIMESTEPS):
#         for filter in range(0, filters):
#             for feature in range(0, right_features):
#                 rightarray[t][right_features*filter + feature] = flat_pooled_R[filter][t][feature]
#
#
# ################# make datasets
#     ds = load_subject(arg_dict["subject"], arg_dict["ROI"], None)
#     if(half <2):
#         starting_t = 0
#     else:
#         starting_t = TIMESTEPS
#     ending_t = starting_t + TIMESTEPS
#
#     temp_targets = ds.targets[starting_t:ending_t]
#     log_file.write("temp_targets has length "+str(len(temp_targets))+"\n")
#
#     temp_chunks = ds.chunks[starting_t:ending_t]
#     log_file.write("temp_chunks has length "+str(len(temp_chunks))+"\n")
#     leftds = dataset_wizard(samples=leftarray, targets=temp_targets,chunks=temp_chunks)
#     rightds = dataset_wizard(samples=rightarray, targets=temp_targets, chunks=temp_chunks)
#     #bilateralds = dataset_wizard(samples=bilateralarray, targets=ds.targets, chunks=ds.chunks)
#
#     left_transformed = open(PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"] + "/half"+str(half)+"_transformed_lh.p", "wb")
#     right_transformed = open(PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"] + "/half"+str(half)+"_transformed_rh.p", "wb")
#     #bi_transformed = open(PATH+"seanfiles/"+arg_dict["subject"]+"/"+arg_dict["ROI"]+"/transformed_lrh.p", "wb")
#
#     pickle.dump(leftds, left_transformed)
#     pickle.dump(rightds, right_transformed)
#     #pickle.dump(bilateralds, bi_transformed)
#
#     log_file.write("saved transformed datasets at "+PATH+"seanfiles/strat"+str(strategy)+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"] +"\n\n")
#     finish_file = open("/isi/music/auditoryimagery2/seanfiles/strat"+str(strategy)+"/finish/half"+str(half)+"_"+str(arg_dict["subject"])+"_"+str(arg_dict["ROI"])+".txt", "w")
#     finish_file.write("finished")
#     finish_file.close()
