import sys
from Constants import *
from temporal_helpers import *
from keras.models import model_from_json
from datetime import datetime
import tensorflow as tf
from random import randint
import numpy as np
from mvpa2.datasets import *
from mvpa2.base.dataset import save


### THIS FILE SHOULD BE RUN WITH PYTHON2 INCLUDING TENSORFLOW-KERAS, PICKLE, AND NUMPY ###
# with tf.device("/cpu:0"):
args = len(sys.argv)
arg_dict = {}

global TIMESTEPS

with tf.device("/cpu:0"):

    AEs_F = 3

    arg_dict["subject"] = str(sys.argv[1])
    arg_dict["ROI"] = str(sys.argv[2])
    strategy = str(sys.argv[3])
    arg_dict["strategy"] = strategy

    iteration = arg_dict["subject"] + "_" + arg_dict["ROI"] + "_strat" + str(strategy)

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
    #which hyperparameter configuration are we using?
    arg_dict["temporal_window1"] = 6

    arg_dict["spatial_window1"] = 16

    arg_dict["num_primary_filters"] = 14

    arg_dict["pooling_cube"] = [2, 2, 2]

    log_file = open(LOGS_PATH+"/"+str(iteration)+".txt","w")

    # ADD ANY OTHER HYPERPARAMETERS THAT WE WANT TO TRACK
    log_file.write("-------- HYPERPARAMETERS ---------\n\n")
    for key in arg_dict:
        log_file.write(str(key) + " :    " + str(arg_dict[key]) + "\n")
    log_file.write("Z_T_Primary :    " + str(ZERO_THRESHOLD_PRIMARY) + "\n")

    log_file.write("\n\n--------- Training Begins --------")

    #all unlabelled samples used to train autoencoder
    if strategy == 0:
        arg_dict["half"] = 0
        TIMESTEPS = 1864
        arg_dict["TIMESTEPS"] = TIMESTEPS
        arg_dict["start_step"] = 0

        strat_samples, new_steps, strat_targets, strat_steps, labelled_steps = make_StratDS(arg_dict, log_file)
        arg_dict["strat_samples"] = strat_samples
        arg_dict["strat_steps"] = strat_steps  # flat number of timesteps

        arg_dict["new_steps"] = new_steps  # list of timesteps
        arg_dict["strat_targets"] = strat_targets
        arg_dict["labelled_steps"] = labelled_steps

        log_file.write("Training primary AE for " + "strategy 0... \n\n")
        best_loss = -1
        best_ae = None
        ae_list = []

        # 12 is arbitrary, we just want to account for failure/underperformance due to random initializations
        for j in range(0, 12):
            ae, window_samples = primary_ae(arg_dict, log_file, "primary")

            ae_list.append(ae)
            log_file.write("AE number " + str(j) + " ended with a training loss of " + str(current_loss[0]) + "\n")
            # current_loss is defined and updated within temporal_helpers.py
            # it's a global variable and is used in the LossCallback function.
            if (best_loss == -1) or (current_loss[0] < best_loss):
                best_loss = current_loss[0]
                best_ae = ae

        print("converting to json..\n")
        model_json = best_ae.to_json()

        with open(ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/" + "strat" + str(
                arg_dict["strategy"]) + "_primary_ae.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        best_ae.save_weights(
            ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/strat" + str(
                strategy) + "_primary_ae.h5")
        log_file.write("Saved model to disk")

        # load json and create model just to make sure it was saved correctly
        jsonpath = ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/strat" + str(
            strategy) + "_primary_ae.json"
        log_file.write("jsonpath is " + str(jsonpath))
        json_file = open(jsonpath, 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(
            ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/strat" + str(
                strategy) + "_primary_ae.h5")
        log_file.write("Loaded model from disk")
        # probably redundant
        TIMESTEPS = 932

        # ############### left side convolution
        log_file.write("Performing primary convolution for left side... \n\n")
        arg_dict["ROI"] = arg_dict["ROI"] + "L"
        flat_pooled_L, left_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None,
                                                        None)
        #
        #
        log_file.write("Saving first block... \n\n")
        arg_dict["ROI"] = arg_dict["ROI"] + "L"
        #
        save_block(arg_dict, flat_pooled_L, None, "primary", log_file, iteration)
        arg_dict["ROI"] = arg_dict["ROI"][:-1]

        log_file.write("first block left side saved")
        #
        filters = len(flat_pooled_L)
        left_features = len(flat_pooled_L[0][0])

        log_file.write("left features is " + str(left_features) + "\n")
        #
        leftarray = np.zeros((TIMESTEPS, left_features * filters))
        log_file.write("leftarray shape is " + str(leftarray.shape))
        #
        for t in range(0, TIMESTEPS):
            for filter in range(0, filters):
                for feature in range(0, left_features):
                    leftarray[t][left_features * filter + feature] = flat_pooled_L[filter][t][feature]
        #
        ################## right side convolution
        log_file.write("Performing primary convolution for right side, the arg_dict value is " + str(
            arg_dict["ROI"]) + "... \n\n")
        arg_dict["ROI"] = arg_dict["ROI"] + "R"
        flat_pooled_R, right_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None,
                                                         None)

        log_file.write("Saving first block... \n\n")
        arg_dict["ROI"] = arg_dict["ROI"] + "R"

        save_block(arg_dict, flat_pooled_R, None, "primary", log_file, iteration)
        arg_dict["ROI"] = arg_dict["ROI"][:-1]

        log_file.write("first block left side saved")

        filters = len(flat_pooled_R)
        right_features = len(flat_pooled_R[0][0])

        log_file.write("right features is " + str(right_features) + "\n")

        rightarray = np.zeros((TIMESTEPS, right_features * filters))
        log_file.write("rightarray shape is " + str(rightarray.shape))

        for t in range(0, TIMESTEPS):
            for filter in range(0, filters):
                for feature in range(0, right_features):
                    rightarray[t][right_features * filter + feature] = flat_pooled_R[filter][t][feature]
        #
        #
        # ################# make datasets
        ds = load_subject(arg_dict["subject"], arg_dict["ROI"], None)
        starting_t = 0

        ending_t = TIMESTEPS

        temp_targets = ds.targets[starting_t:ending_t]
        log_file.write("temp_targets has length " + str(len(temp_targets)) + "\n")

        temp_chunks = ds.chunks[starting_t:ending_t]
        log_file.write("temp_chunks has length " + str(len(temp_chunks)) + "\n")
        leftds = dataset_wizard(samples=leftarray, targets=temp_targets, chunks=temp_chunks)
        rightds = dataset_wizard(samples=rightarray, targets=temp_targets, chunks=temp_chunks)

        left_transformed = open(
            ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/" + "strat" + str(
                strategy) + "_transformed_lh.p", "wb")
        right_transformed = open(
            ROI_PATH + "/" + str(arg_dict["subject"]) + "/" + str(arg_dict["ROI"]) + "/strat" + str(
                strategy) + "_transformed_rh.p", "wb")

        pickle.dump(leftds, left_transformed)
        pickle.dump(rightds, right_transformed)

        left_transformed.close()
        right_transformed.close()

        log_file.write("saved transformed datasets at " + ROI_PATH + "/" + arg_dict["subject"] + "/" + arg_dict[
            "ROI"] + "\n\n")

    #half-and-half non-circular strategy
    #half refers to runs 1-4 or 5-8, not which half of the brain
    else:
        half_dict = {1: "runs_1_through_4", 2: "runs_5_through_8"}
        arg_dict["half_dict"] = half_dict
        for half in [1, 2]:
            arg_dict["half"]=half
            TIMESTEPS=932
            arg_dict["TIMESTEPS"]=TIMESTEPS
            arg_dict["start_step"] = (half-1)*TIMESTEPS

            strat_samples, new_steps, strat_targets, strat_steps, labelled_steps = make_StratDS(arg_dict, log_file)
            arg_dict["strat_samples"]=strat_samples
            arg_dict["strat_steps"]=strat_steps #flat number of timesteps

            arg_dict["new_steps"]=new_steps #list of timesteps
            arg_dict["strat_targets"]=strat_targets
            arg_dict["labelled_steps"]=labelled_steps

            log_file.write("Training primary AE for "+half_dict[half]+"... \n\n")
            best_loss = -1
            best_ae = None
            ae_list = []

            #12 is arbitrary, we just want to account for failure/underperformance due to random initializations
            for j in range(0,12):
                ae, window_samples = primary_ae(arg_dict, log_file, "primary")

                ae_list.append(ae)
                log_file.write("AE number "+str(j)+" ended with a training loss of "+str(current_loss[0])+"\n")
                #current_loss is defined and updated within temporal_helpers.py
                #it's a global variable and is used in the LossCallback function.
                if (best_loss == -1) or (current_loss[0] < best_loss):
                    best_loss = current_loss[0]
                    best_ae = ae


            print("converting to json..\n")
            model_json = best_ae.to_json()

            with open(ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"])+"/"+"strat"+str(arg_dict["strategy"])+"_"+half_dict[half]+"_primary_ae.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            best_ae.save_weights(ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"])+"/strat"+str(strategy)+"_"+half_dict[half]+"_primary_ae.h5")
            log_file.write("Saved model to disk")

            #load json and create model just to make sure it was saved correctly
            jsonpath = ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"])+"/strat"+str(strategy)+"_"+half_dict[half]+"_primary_ae.json"
            log_file.write("jsonpath is "+str(jsonpath))
            json_file = open(jsonpath, 'r')
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)

            # load weights into new model
            loaded_model.load_weights(ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"])+"/strat"+str(strategy)+"_"+half_dict[half]+"_primary_ae.h5")
            log_file.write("Loaded model from disk")
            #probably redundant
            TIMESTEPS = 932


# ############### left side convolution
            log_file.write("Performing primary convolution for left side... \n\n")
            arg_dict["ROI"] = arg_dict["ROI"]+"L"
            flat_pooled_L, left_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None, None)
        #
        #
            log_file.write("Saving first block... \n\n")
            arg_dict["ROI"] = arg_dict["ROI"]+"L"
        #
            save_block(arg_dict, flat_pooled_L, None, "primary", log_file, iteration)
            arg_dict["ROI"] = arg_dict["ROI"][:-1]

            log_file.write("first block left side saved")
        #
            filters = len(flat_pooled_L)
            left_features = len(flat_pooled_L[0][0])

            log_file.write("left features is "+str(left_features)+"\n")
        #
            leftarray = np.zeros((TIMESTEPS, left_features*filters))
            log_file.write("leftarray shape is "+str(leftarray.shape))
        #
            for t in range(0, TIMESTEPS):
                for filter in range(0, filters):
                    for feature in range(0, left_features):

                        leftarray[t][left_features*filter + feature] = flat_pooled_L[filter][t][feature]
#
################## right side convolution
            log_file.write("Performing primary convolution for right side, the arg_dict value is "+str(arg_dict["ROI"])+"... \n\n")
            arg_dict["ROI"] = arg_dict["ROI"]+"R"
            flat_pooled_R, right_count = primary_convolution(loaded_model, arg_dict, log_file, "primary", None, None, None)


            log_file.write("Saving first block... \n\n")
            arg_dict["ROI"] = arg_dict["ROI"]+"R"

            save_block(arg_dict, flat_pooled_R, None, "primary", log_file, iteration)
            arg_dict["ROI"] = arg_dict["ROI"][:-1]

            log_file.write("first block left side saved")

            filters = len(flat_pooled_R)
            right_features = len(flat_pooled_R[0][0])

            log_file.write("right features is "+str(right_features)+"\n")

            rightarray = np.zeros((TIMESTEPS, right_features*filters))
            log_file.write("rightarray shape is "+str(rightarray.shape))

            for t in range(0, TIMESTEPS):
                for filter in range(0, filters):
                    for feature in range(0, right_features):
                        rightarray[t][right_features*filter + feature] = flat_pooled_R[filter][t][feature]
#
#
# ################# make datasets
            ds = load_subject(arg_dict["subject"], arg_dict["ROI"], None)
            if(half == 1):
                starting_t = 0
            else:
                starting_t = TIMESTEPS
            ending_t = starting_t + TIMESTEPS

            temp_targets = ds.targets[starting_t:ending_t]
            log_file.write("temp_targets has length "+str(len(temp_targets))+"\n")

            temp_chunks = ds.chunks[starting_t:ending_t]
            log_file.write("temp_chunks has length "+str(len(temp_chunks))+"\n")
            leftds = dataset_wizard(samples=leftarray, targets=temp_targets,chunks=temp_chunks)
            rightds = dataset_wizard(samples=rightarray, targets=temp_targets, chunks=temp_chunks)

            left_transformed = open(ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"]) + "/" +"strat"+str(strategy)+"_"+half_dict[half]+"_transformed_lh.p", "wb")
            right_transformed = open(ROI_PATH+"/"+str(arg_dict["subject"])+"/"+str(arg_dict["ROI"]) + "/strat"+str(strategy)+"_"+half_dict[half]+"_transformed_rh.p", "wb")

            pickle.dump(leftds, left_transformed)
            pickle.dump(rightds, right_transformed)

            left_transformed.close()
            right_transformed.close()


            log_file.write("saved transformed datasets at "+ROI_PATH+"/"+arg_dict["subject"]+"/"+arg_dict["ROI"] +"\n\n")

