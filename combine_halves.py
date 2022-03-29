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

subject = str(sys.argv[1])
ROIs = [1001, 1012, 1019, 1020, 1024, 1027, 1030,  1031, 1034, 1035]
strategy=1 #the half-and-half strategy
directory = LOGS_PATH
log_file = open(directory+"/"+subject+"_combine_log.txt","w")
half_dict = {1: "runs_1_through_4", 2: "runs_5_through_8"}

TIMESTEPS = 932
for roi in ROIs:
    log_file.write("ROI "+str(roi)+":\n")
    log_file.write("Loading halves 1 and 2 for left-handed brain...\n")
    data_p1 = open(ROI_PATH + subject + "/" + str(roi) + "/"+ "strat"+str(strategy)+"_"+half_dict[1]+"_transformed_lh.p", "rb")
    data_p2 = open(ROI_PATH + subject + "/" + str(roi) + "/"+ "strat"+str(strategy)+"_"+half_dict[2]+"_transformed_lh.p", "rb")
    log_file.write("Loading halves 1 and 2 for right-handed brain...\n")

    data_p3 = open(ROI_PATH + subject + "/" + str(roi) + "/"+ "strat"+str(strategy)+"_"+half_dict[1]+"_transformed_rh.p", "rb")
    data_p4 = open(ROI_PATH + subject + "/" + str(roi) + "/"+ "strat"+str(strategy)+"_"+half_dict[2]+"_transformed_rh.p", "rb")

    left1 = pickle.load(data_p1)
    left2 = pickle.load(data_p2)
    log_file.write("Left hand sample shapes are "+str(left1.samples.shape)+" and "+str(left2.samples.shape))
    log_file.write("Left hand target shapes are "+str(left1.targets.shape)+" and "+str(left2.targets.shape))
    log_file.write("Left hand chunks shapes are "+str(left1.chunks.shape)+" and "+str(left2.chunks.shape))

    num_features1L = left1.samples.shape[1]
    num_features2L= left2.samples.shape[1]
    smaller = None
    bigger = None
    if(num_features1L>num_features2L):
        # log_file.write("feature count mistmatch, quitting...\n")
        # exit(0)
        smaller = left2
        small_num = left2.samples.shape[1]
        temp = np.zeros((1864,small_num))
        for i in range(0,1864):
            for j in range(0,small_num):
                temp[i][j] = left1.samples[i][j]

        left1.samples = temp
        print("left 1 now has shape "+str(left1.samples.shape))
        print("and left 2 has shape "+str(left2.samples.shape))

    elif(num_features1L<num_features2L):
        smaller = left1
        small_num = left1.samples.shape[1]
        temp = np.zeros((1864,small_num))
        for i in range(0,1864):
            for j in range(0,small_num):
                temp[i][j] = left2.samples[i][j]

        left2.samples = temp
        print("left 2 now has shape "+str(left2.samples.shape))
        print("and left 2 has shape "+str(left1.samples.shape))
    num_features1L = left1.samples.shape[1]
    num_features2L= left2.samples.shape[1]
    l_feats = num_features1L
    comb_samples = np.zeros((1864, l_feats))
    comb_targets = np.zeros(1864,)
    comb_chunks = np.zeros(1864,)
    for i in range(0,932):
        for j in range(0, l_feats):
            comb_samples[i,j] = left1.samples[i,j]
        comb_targets[i] = left1.targets[i]
        comb_chunks[i] = left1.chunks[i]
    for i in range(0,932):
        for j in range(0, l_feats):
            comb_samples[i+932,j] = left2.samples[i,j]
        comb_targets[i+932] = left2.targets[i]
        comb_chunks[i+932] = left2.chunks[i]

    leftds = dataset_wizard(samples=comb_samples, targets=comb_targets, chunks=comb_chunks)
    data_p1.close()
    data_p2.close()

    log_file.write("Left combined dataset has shapes "+str(leftds.samples.shape)+", "+str(leftds.targets.shape)+", and "+str(leftds.chunks.shape)+"\n\n\n")


    ########### RIGHT HAND SIDE ######################
    right1 = pickle.load(data_p3)
    right2 = pickle.load(data_p4)
    log_file.write("Right hand sample shapes are " + str(right1.samples.shape) + " and " + str(right2.samples.shape))
    log_file.write("Right hand target shapes are " + str(right1.targets.shape) + " and " + str(right2.targets.shape))
    log_file.write("Right hand chunks shapes are " + str(right1.chunks.shape) + " and " + str(right2.chunks.shape))

    num_features1R = right1.samples.shape[1]
    num_features2R = right2.samples.shape[1]

    if(num_features1R>num_features2R):
        # log_file.write("feature count mistmatch, quitting...\n")
        # exit(0)
        small_num = right2.samples.shape[1]
        temp = np.zeros((1864,small_num))
        for i in range(0,1864):
            for j in range(0,small_num):
                temp[i][j] = right1.samples[i][j]

        right1.samples = temp
        print("right 1 now has shape "+str(right1.samples.shape))
        print("and right 2 has shape "+str(right2.samples.shape))
    elif(num_features1L<num_features2L):
        small_num = right1.samples.shape[1]
        temp = np.zeros((1864,small_num))
        for i in range(0,1864):
            for j in range(0,small_num):
                temp[i][j] = right2.samples[i][j]
        right2.samples = temp
        print("right 2 now has shape "+str(right2.samples.shape))
        print("and right 2 has shape "+str(right1.samples.shape))
    num_features1R = right1.samples.shape[1]
    num_features2R = right2.samples.shape[1]

    r_feats = num_features1R
    comb_samples = np.zeros((1864, r_feats))
    comb_targets = np.zeros(1864, )
    comb_chunks = np.zeros(1864, )
    for i in range(0, 932):
        for j in range(0, r_feats):
            comb_samples[i, j] = right1.samples[i, j]
        comb_targets[i] = right1.targets[i]
        comb_chunks[i] = right1.chunks[i]
    for i in range(0, 932):
        for j in range(0, r_feats):
            comb_samples[i + 932, j] = right2.samples[i, j]
        comb_targets[i + 932] = right2.targets[i]
        comb_chunks[i + 932] = right2.chunks[i]

    rightds = dataset_wizard(samples=comb_samples, targets=comb_targets, chunks=comb_chunks)
    data_p3.close()
    data_p4.close()

    log_file.write("Right combined dataset has shapes " + str(rightds.samples.shape) + ", " + str(
        rightds.targets.shape) + ", and " + str(rightds.chunks.shape) + "\n\n")


    left_transformed = open(
        ROI_PATH +"/" + subject + "/" + str(roi) + "/strat1_transformed_lh.p", "wb")
    right_transformed = open(
        ROI_PATH +"/" + subject + "/" + str(roi) + "/strat1_transformed_rh.p", "wb")

    pickle.dump(leftds, left_transformed)
    pickle.dump(rightds, right_transformed)

    left_transformed.close()
    right_transformed.close()

    log_file.write("Combined datasets saved. Next ROI...\n\n\n")
