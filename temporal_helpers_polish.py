### General imports ###
from random import randint
from Constants import ZERO_THRESHOLD_PRIMARY, ZERO_THRESHOLD_SECONDARY, P_EPOCHS, S_EPOCHS, P_KERNEL_REGULARIZER, P_ACTIVITY_REGULARIZER, S_ACTIVITY_REGULARIZER, S_BATCH_SIZE, S_KERNEL_REGULARIZER, P_BATCH_SIZE, ACCESSIONS, PATH
#from stratDS import StratDS
import numpy
import pickle
from math import tanh
import tensorflow as tf
import keras
### Keras imports ###
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D
from keras import regularizers
from keras import Model
from pickle_data_new import *

TEMPORAL_WINDOW2 = "NULL"
NUM_SECONDARY_FILTERS = "NULL"
SPATIAL_WINDOW2 = "NULL"
POOLING_CUBE = "NULL"
#returns the reference to the trained keras model

current_loss = [-1]


class LossCallback(keras.callbacks.Callback):
    global current_loss
    def on_epoch_end(self, epoch, logs={}):
        global current_loss
        #print("according to the log, loss is "+str(logs.get('loss')))
        #print("according to the log, val loss is "+str(logs.get('val_loss')))
        current_loss[0] = logs.get('val_loss')

def primary_ae(arg_dict, log_file, layer):
    global TEMPORAL_WINDOW2, NUM_SECONDARY_FILTERS, SPATIAL_WINDOW2, POOLING_CUBE
    loss_callback = LossCallback()

    if layer=="primary":
        half = arg_dict["half"]
        TIMESTEPS = arg_dict["TIMESTEPS"]
        subject = arg_dict["subject"]
        ROI = arg_dict["ROI"]
        ds = load_subject(subject, ROI, log_file)
        if(arg_dict["strategy"]!=0):
            samples=arg_dict["strat_samples"]
            labelled_steps = arg_dict["labelled_steps"]
        else:
            samples=ds.samples
        num_samples = samples.shape[1]
        if num_samples < 100:
            arg_dict["pooling_cube"] = [1,1,1]
            log_file.write("There were too few samples for proper pooling, so the cube was redued to a single fv")
        log_file.write("Immediately after loading subject, samples has shape "+str(samples.shape)+"\n")
        #targets = ds.targets
        #chunks = ds.chunks
        TEMPORAL_WINDOW2 = arg_dict["temporal_window2"]
        NUM_SECONDARY_FILTERS = arg_dict["num_secondary_filters"]
        SPATIAL_WINDOW2 = arg_dict["spatial_window2"]
        POOLING_CUBE = arg_dict["pooling_cube"]


        num_epochs = P_EPOCHS

        num_features = samples.shape[1]

        window_size = arg_dict["temporal_window1"]
        timesteps = arg_dict["strat_steps"]
        hidden_dim = arg_dict["num_primary_filters"]

    elif (layer=="secondary"):
        samples = arg_dict
        samples = numpy.array(samples)
        log_file.write("During secondary ae, samples has shape "+str(samples.shape)+"\n")
        num_features = len(samples[0])
        #window_size = TEMPORAL_WINDOW2
        window_size = TEMPORAL_WINDOW2
        timesteps = TIMESTEPS
        #hidden_dim = NUM_SECONDARY_FILTERS
        hidden_dim = NUM_SECONDARY_FILTERS

        num_epochs = S_EPOCHS

    # We'll train our autoencoder on time-samples of size window_size. window_samples is the collection of them all.
        # Do we potentially want to modify this collection?
    window_samples = []
    if arg_dict["half"] == 2:
        spacer = timesteps
    else:
        spacer = 0
    log_file.write("samples has length "+str(len(samples))+"\n")
    if half ==  0:
        half = 1 #hacky so the for loop runs correctly in that case
    overlap_cnt = 0
    log_file.write("gathering window samples from t="+str((half-1)*timesteps)+" to t="+str(timesteps*half)+"\n")
    for i in range( spacer, spacer+timesteps - window_size):
        #print("timestep "+str(i)+"\n")

        for j in range(0, num_features):
            temp = []
            for k in range(0, window_size):
                temp.append(samples[i + k][j])
            choice = randint(1,6)
            if choice == 6:
                windowsteps = range(i,i+window_size)
                #if(any((item+spacer) in labelled_steps for item in windowsteps)):
                if False:
                    overlap_cnt=overlap_cnt+1
                    #log_file.write("windowsteps "+str(windowsteps)+" overlapped with labelled_steps. \n")
                else:
                    window_samples.append(numpy.array(temp))

    window_samples = numpy.array(window_samples)
    samples = numpy.array(samples)
    log_file.write("There were "+str(overlap_cnt)+" overlaps\n")
    log_file.write("window samples has shape "+str(window_samples.shape)+" and samples has shape "+str(samples.shape)+"\n")
    #We are ready to train the autoencoder
    ### TRAIN PRIMARY AUTOENCODER ###

    log_file.write("Samples tensor has shape " + str(samples.shape) + "\n")
    log_file.write("Window samples tensor has shape " + str(window_samples.shape) + "\n")


    input = Input(shape=(window_size,))

    encoded = Dense(hidden_dim, activation='linear'
                    , kernel_regularizer=regularizers.l1(P_KERNEL_REGULARIZER)
                    , activity_regularizer=regularizers.l1(P_ACTIVITY_REGULARIZER)
                    , name="encoder"
                    )(input)
    inter = Activation('relu', name='inter')(encoded)

    decoded = Dense(window_size, activation='relu')(inter)

    autoencoder = Model(input, decoded)

    autoencoder.compile(optimizer='adamax', loss='mse',
                        metrics=['cosine_proximity'])

    log_file.write("Training "+layer+" autoencoder... \n\n")
    autoencoder.fit(window_samples, window_samples, epochs=num_epochs,
                    batch_size=P_BATCH_SIZE,
                    shuffle=True,
                    verbose=2,
                    validation_split=0.1,
                    callbacks=[loss_callback]
                    )

    return autoencoder,window_samples





def primary_convolution(ae, arg_dict, log_file, layer, spatial_info_l, spatial_info_r, lcount):
    if (layer=="primary"):
        subject = arg_dict["subject"]
        ROI = arg_dict["ROI"]
        num_filters = arg_dict["num_primary_filters"]
        temporal_window = arg_dict["temporal_window1"]
        spatial_window = arg_dict["spatial_window1"]
        pooling_cube = arg_dict["pooling_cube"]
        left_count = -1
        TIMESTEPS = arg_dict["TIMESTEPS"]
        half = arg_dict["half"]
        directory = PATH+"seanfiles/"+str(subject)+"/"+str(ROI)
        if half == 0:
            start_t=0
            end_t = TIMESTEPS
        if half == 1:
            start_t=0
            end_t=TIMESTEPS
        if half == 2:
            start_t=TIMESTEPS
            end_t=TIMESTEPS*2
        ############ LOAD DESIRED DATA ############
        if (ROI[-1]=="L"):
            pre_ds = get_subject_ds(subject)
            samples = pre_ds.samples
            num_samples = samples.shape[1]
            if num_samples < 30:
                arg_dict["pooling_cube"] = [1, 1, 1]
                pooling_cube = arg_dict["pooling_cube"]
                log_file.write("There were too few samples for proper pooling, so the cube was redued to a single fv")
            roi = arg_dict["ROI"][:-1] #cut off the L from the end of the ROI name
            roi = int(roi)
            log_file.write("loading left side of ROI, the value is "+str(roi)+"\n")
            rois = [roi]
            ds= mask_subject_ds(pre_ds, subject, rois)
            log_file.write("left side has this many samples: "+str(ds.samples.shape)+"\n")
        elif (ROI[-1]=="R"):

            pre_ds = get_subject_ds(subject)
            samples = pre_ds.samples
            num_samples = samples.shape[1]
            if num_samples < 30:
                arg_dict["pooling_cube"] = [1, 1, 1]
                pooling_cube = arg_dict["pooling_cube"]
                log_file.write("There were too few samples for proper pooling, so the cube was redued to a single fv")
            roi = arg_dict["ROI"][:-1] #cut off the R from the end of the ROI name
            roi = int(roi)+1000
            log_file.write("loading right side of ROI, the value is "+str(roi)+"\n")
            rois = [roi]
            ds= mask_subject_ds(pre_ds, subject, rois)
            log_file.write("right side has this many samples: "+str(ds.samples.shape)+"\n")
        elif (ROI[-1]=="B"):
            ds = load_subject(subject, ROI[:-1], log_file)
            log_file.write("loading rbilteral ROI \n")
        else:
            ds = None
            log_file.write("The lateral/bilateral side was not labeled when passed to primary convolution. \n\n")



        arg_dict["ROI"] = arg_dict["ROI"][:-1]
        samples = ds.samples[start_t:end_t]
        orig_data=ds.O[start_t:end_t]
        voxel_indices = ds.fa.voxel_indices
        num_features = samples.shape[1]
        log_file.write("orig_data has shape "+str(orig_data.shape)+"\n\n")
        #log_file.write("in primary convolution, the values at t=102 are "+str(samples[102])+"\n\n")

    elif (layer=="secondary"):
        samples = arg_dict
        num_features = len(arg_dict[0])
        if(num_features==1):
            num_features = len(arg_dict[0][0])
            log_file.write("there was an extra dimension on the list, so num features is "+str(num_features))

        #num_filters = NUM_SECONDARY_FILTERS
        num_filters = NUM_SECONDARY_FILTERS
        #temporal_window = TEMPORAL_WINDOW2
        temporal_window = TEMPORAL_WINDOW2
        #spatial_window = SPATIAL_WINDOW2
        spatial_window = SPATIAL_WINDOW2
        #pooling_cube = POOLING_CUBE
        pooling_cube = POOLING_CUBE

    else:
        log_file.write(layer + " convolution received unknown layer value, quitting... \n")
        log_file.write(layer + " convolution received unknown layer value, quitting... \n")
        return -1

    ############ Convolve along the time axis, using the trained AE neurons as filters, return value is the tensor of response matrices
    responses = convolve(ae, samples, num_features, num_filters, temporal_window, log_file, arg_dict)
    #log_file.write("in the first response matrix, at time step 102 stuff looks like "+str(responses[1][102])+"\n")
    log_file.write("after the convolve call, responses has shape "+str(responses.shape)+"\n")
    log_file.write("Obtained response tensor of shape "+str(responses.shape)+". \n\n")


    ##  CURRENTLY NOT NEEDED, FUNCTIONALITY WAS PUT INTO pooling() BELOW ##
    ########### Updates orig_data with the encoded values, in place, so nothing is returned
    #### The terms left and right refer to the values of the voxels' first dimension. Thus the physical left hemisphere of the subject is called "right" in this code and vice versa
    #get_3d_encoded(orig_data, samples, responses, num_filters, voxel_indices, directory, log_file)

    ########## Max-pools in the 3d space and applies tanh to each result (according the paper), then returns the result
    if(layer=="primary"):
        # ENABLE FOR SPLIT/TWO LAYER FUNCTIONALITY
        #left_3d_pooled, right_3d_pooled = pooling(orig_data, responses, voxel_indices, num_filters, spatial_window, pooling_cube, log_file)
        #log_file.write("testing after pooling: "+str(left_3d_pooled[1][1]))
        #response_len = len(responses[3][0])
        #test_count = 0
        
        # for q in range(0, response_len):
        #     if(responses[3][0][q]==0):
        #         test_count += 1
        #print("test count is "+str(test_count)+"\n")
        # for r in range(1, 1864):
        #     new_count = 0
        #     for s in range(0, response_len):
        #         if(responses[3][r][s]==0):
        #             new_count+=1
        #     if(new_count!=test_count):
        #         print("at time "+str(r)+" the count was "+str(new_count)+"\n")
        #     if(r==1):
        #         print("just to be safe, at time 1 the new count was "+str(new_count)+"\n")

    
        flat_pooled, left_count = pooling(orig_data, responses, voxel_indices, num_filters, spatial_window, pooling_cube, log_file, arg_dict)

        log_file.write("testing t=102 after pooling: "+str(flat_pooled[0][102]))

    else:
        left_3d_pooled, right_3d_pooled = secondary_pooling(responses, num_filters, spatial_window, pooling_cube, spatial_info_l, spatial_info_r, lcount, log_file)

    # ENABLE FOR SPLIT/TWO LAYER FUNCTIONALITY
    #return left_3d_pooled, right_3d_pooled

    return flat_pooled, left_count

def secondary_ae(arg_dict, secondary_flat, log_file):
    #numpy mumbo jumbo to get the right shape/format
    autoencoders = []
    num_primary_filters = arg_dict["num_primary_filters"]
    log_file.write("length of secondary flat should be "+str(num_primary_filters)+" : "+str(len(secondary_flat))+"\n")
    for response in range(0, num_primary_filters):
        # ae_set = []
        # for filter in range(0, NUM_SECONDARY_FILTERS):
        #     #get a new set of autoencoders for each response matrix
        #     ae, window = primary_ae(secondary_flat[response], log_file, "secondary")
        #     ae_set.append(ae)
        # autoencoders.append(ae_set)


        #get a new autoencoder for each response matrix
        ae, window = primary_ae(secondary_flat[response], log_file, "secondary")
        autoencoders.append(ae)

    return autoencoders

def secondary_convolution(autoencoders, arg_dict, spatial_info_l, spatial_info_r, secondary_flat, lcount, log_file):

    # remember that responses has shape (1, NUM_PRIMARY_FILTERS, 1864, 2313//TEMPORAL_WINDOW1)
    secondary_responses_l = []
    secondary_responses_r = []

    for i in range(0, len(autoencoders)):
        left, right = primary_convolution(autoencoders[i], secondary_flat[i], log_file, "secondary", spatial_info_l, spatial_info_r, lcount)
        secondary_responses_l.append(left)
        secondary_responses_r.append(right)

    return secondary_responses_l, secondary_responses_r


def secondary_pooling(responses, num_filters, spatial_window, pooling_cube, spatial_info_l, spatial_info_r, lcount, log_file):
    response_matrices = responses
    #if numpy mumbo jumbo has an extra dimension at the front, just get rid of it
    if(len(responses)==1):
        response_matrices = responses[0]

    pooled_final_l = []
    pooled_final_r = []



    log_file.write("The dimensions for the left secondary padded cube are "+str(spatial_info_l)+"\n")
    log_file.write("The dimensions for the right secondary padded cube are "+str(spatial_info_r)+"\n\n\n")


    #effectively the same as i in range 0, num_filters
    for filter in range(0, len(response_matrices)):
        timeseries_l = []
        timeseries_r = []

        for t in range(0, TIMESTEPS):
        #recall that our 3d left and right are approximately cubes in python lists, so put them in zero padded numpy arrays to be actual cubes
            left = get_numpy_cube(response_matrices[filter][t], spatial_info_l, 0)
            right = get_numpy_cube(response_matrices[filter][t], spatial_info_r, lcount)

            #we can lose any notion of 3d space now, these will just return flattened lists
            left_pooled = pooling_3d(left, pooling_cube)
            right_pooled = pooling_3d(right, pooling_cube)

            timeseries_l.append(left_pooled)
            timeseries_r.append(right_pooled)

        pooled_final_l.append(timeseries_l)
        pooled_final_r.append(timeseries_r)

    return pooled_final_l, pooled_final_r

#starting index is either 0 or lcount, so that the values for the right side cube are taken starting at index lcount in the flattened array,
    # since the flattened array is left + right
def get_numpy_cube(flat_values, spatial_info, starting_index):
    x_dim = spatial_info[0]
    y_dim = spatial_info[1]
    z_dim = spatial_info[2]
    y_lengths = spatial_info[3]
    z_lengths = spatial_info[4]

#    print("xdim is "+str(x_dim))
#    print("ydim is "+str(y_dim))
#    print("zdim is "+str(z_dim))
#    print("length of flat values is "+str(len(flat_values)))
#    print("There are "+str(len(y_lengths))+" many y lengths\n")
#    print("There are "+str(len(z_lengths))+" many z lengths\n")


    cube = numpy.zeros((x_dim, y_dim, z_dim))
    count = 0
    z_lengths_index = 0

    for x in range(0, x_dim):
        #how far do we go in the y direction at that x value?
 #       print("y is about to range from 0 to "+str(y_lengths[x])+"\n")
        for y in range(0, y_lengths[x]):
            #how far do we go in the z direction at that x value?
  #          print("z lengths index is "+str(z_lengths_index)+"\n")
   #         print("z is about to range from 0 to" + str(z_lengths[z_lengths_index])+" \n")
            for z in range(0, z_lengths[z_lengths_index]):

                cube[x][y][z] = flat_values[count + starting_index]
                count+=1
            #remember our position in the z_lengths list
            z_lengths_index += 1


    return cube

def pooling_3d(cube, pooling_cube):
    pooled_values = []

    x_stride = pooling_cube[0]
    y_stride = pooling_cube[1]
    z_stride = pooling_cube[2]

    x_len = len(cube)
    y_len = len(cube[0])
    z_len = len(cube[0][0])

    for x in range(0, x_len, x_stride):
        for y in range(0, y_len, y_stride):
            for z in range(0, z_len, z_stride):
                ##### set initial c_x,y,z values i think? So they only change for the one iteration where we passed an if statement below

                #if there aren't enough values in the cube to use the default pooling cube, resize it
                if (x+x_stride > x_len):
                    c_x = x_len - x
                else:
                    c_x = x_stride

                if (y+y_stride > y_len):
                    c_y = y_len - y
                else:
                    c_y = y_stride

                if (z+z_stride > z_len):
                    c_z = z_len - z
                else:
                    c_z = z_stride

                sub_cube = get_cube(cube, -1, x, y, z, c_x, c_y, c_z)

                if(zero_check(sub_cube, c_x, c_y, c_z, "secondary")):
                    pooled_values.append(tanh(numpy.amax(sub_cube)))

    return pooled_values



#layer is either 'primary' or 'secondary'
def save_block(arg_dict, left, right, layer, log_file, iteration):
    subject = arg_dict["subject"]
    side = arg_dict["ROI"][-1]
    ROI = arg_dict["ROI"][:-1]
    directory = "/isi/music/auditoryimagery2/seanfiles/strat"+str(arg_dict["strategy"])\
                +"/"+str(subject)+"/"+str(ROI)+"/"

    # ENABLE FOR SPLIT/TWO LAYER FUNCTIONALITY
    # log_file.write("Saving left and right sides after" +str(layer)+" block...\n")
    # left_p = open(directory+"/left_"+layer+str(iteration)+".p", "wb")
    # right_p = open(directory+"/right_"+layer+str(iteration)+".p", "wb")
    #
    # pickle.dump(left, left_p)
    # pickle.dump(right, right_p)
    #
    # log_file.write("Saving left and right sides after"+str(layer) +" block for python2...\n")
    # left_p2 = open(directory+"/left_"+layer+str(iteration)+"_py2.p", "wb")
    # right_p2 = open(directory+"/right_"+layer+str(iteration)+"_py2.p", "wb")
    #
    # pickle.dump(left, left_p2, protocol=2)
    # pickle.dump(right, right_p2, protocol=2)

    log_file.write("Saving flat pooled values, these should be concatenated and then made into an mvpa dataset object\n")
    flat_p = open(directory+"half"+str(arg_dict["half"])+"_primary_pooled_"+str(side)+".p","wb")

    pickle.dump(left, flat_p)



#layer is either 'primary' or 'secondary'
def load_block(arg_dict, layer, version, log_file):
    subject = arg_dict["subject"]
    ROI = arg_dict["ROI"]
    directory = "/isi/music/auditoryimagery2/seanfiles/"+str(subject)+"/"+str(ROI)+"/"

    #if we are loading in python3, the filename is just left/right_primary/secondary, so make version the empty string
    if int(version)==3:
        version = ""
    else:
        version = "_py2"
    log_file.write("Loading "+layer+" block of left and right sides...\n")
    left_p = open(directory+"/left_"+layer+version+str(iteration)+".p", "rb")
    right_p = open(directory+"/right_"+layer+version+str(iteration)+".p", "rb")

    left = pickle.load(left_p)
    right = pickle.load(right_p)

    return left, right


def load_subject(subject, ROI, log_file):
    #log_file.write("Loading pickled data... \n\n")

    directory = "/isi/music/auditoryimagery2/seanfiles/"+str(subject)+"/"+str(ROI)
    # s_p = open(directory + "/samples.p", "rb")
    # c_p = open(directory + "/chunks.p", "rb")
    # t_p = open(directory + "/targets.p", "rb")
    #
    # samples = pickle.load(s_p, encoding="bytes")
    # chunks = pickle.load(c_p, encoding="bytes")
    # targets = pickle.load(t_p, encoding="bytes")

    ds_p = open(directory + "/raw_ds.p", "rb")

    ds = pickle.load(ds_p)

    return ds

def convolve(ae, samples, num_features, num_filters, temporal_window, log_file, arg_dict):
    # numpy mumbo jumbo to get the right shape/format
    samples = [samples]
    samples = [samples]
    TIMESTEPS = arg_dict["TIMESTEPS"]
    samples = numpy.array(samples)

    log_file.write("Convolving along time axis with temporal window "+str(temporal_window)+"... \n\n")
    #### convolution architecture #####
    full_input = Input(shape=(1, TIMESTEPS, num_features))

    # convolve along the time dimension
    firstConv = Conv2D(num_filters, (temporal_window, 1), strides=1, padding='same',
                       data_format='channels_first', name='conv1')
    # sequential API
    conv1 = firstConv(full_input)

    firstBlock = Model(inputs=full_input, outputs=conv1)
    # these parameters don't matter since we're not training here. the compile call is just to build the graph.
    firstBlock.compile(optimizer='adamax', loss='mse',
                       metrics=['cosine_proximity'])

    # use loaded model's neurons as filters
    # required numpy mumbo jumbo to make it the right shape even though it already was
    trained_weights = ae.get_layer('encoder').get_weights()
    log_file.write("trained weights is "+str(trained_weights))

    new_weights = []
    for i in range(0, temporal_window):
        a = trained_weights[0][i]  # a is a list
        a = [[a]]
        new_weights.append(a)
    new_weights = numpy.array(new_weights)
    # new_weights is the first item in the final array.

    new_biases = numpy.array(trained_weights[1])
    # new_biases is the second item in the final array

    final_weights = [new_weights, new_biases]
    final_weights = numpy.array(final_weights)
    # print("final_weights has shape "+str(final_weights.shape))

    # finally set the weights to be our trained filters
    firstConv.set_weights(final_weights)

    out = firstBlock.predict(samples)
    out = out[0] #out has an extra dimension for stupid numpy reasons

    return out


######### fills the 3d volume with the values from responses (which is flat)
# def get_3d_encoded(volume, flattened, responses, num_filters, voxel_indices, directory, log_file):
#     #our map for placing the encoded values in "responses" into the 3d volume
#
#     zerocount = 0
#     left_encoded_responses = []
#     right_encoded_responses = []
#
#     #We need to know where to look to find the active voxels, then store those bounds for our loops to use
#     l_x, l_x_min, l_y, l_y_min, l_z, l_z_min, r_x, r_x_min, r_y, r_y_min, r_z, r_z_min = get_voxel_bounds(voxel_indices, log_file)
#
#     for response in range(0, num_filters):
#         nonzerocount = 0
#         # we need to create the encoded version of the original data for each filter/response matrix
#         log_file.write("Overwriting old data with encoded data for response " + str(response)+"\n\n")
#         for t in range(0, TIMESTEPS):
#             count = 0
#             #The list of voxel indices is in the same order as the flattened values, so we can count through "responses" with it
#             for voxel in voxel_indices:
#                 i = voxel[0]
#                 j = voxel[1]
#                 k = voxel[2]
#                 # print("i, j, and k are "+str(i)+", "+str(j)+", "+str(k))
#                 # get the encoded value corresponding to that voxel at that point in time
#                 enc_value = responses[0][response][t][count]
#                 if (enc_value == 0):
#                     zerocount += 1
#                     # print("encoded value of zero: count is "+str(zerocount))
#
#                 # replace that voxel value with the encoded value
#                 volume[t][i][j][k] = enc_value
#
#                 count += 1
#
#
#         log_file.write("slicing left hemi of response " + str(response) + "...")
#         left_hemi = numpy.zeros((TIMESTEPS, l_x, l_y, l_z))
#
#         for t in range(0, TIMESTEPS):
#             for x in range(0, l_x):
#                 for y in range(0, l_y):
#                     for z in range(0, l_z):
#                         value = volume[t][l_x_min + x][l_y_min + y][l_z_min + z]
#                         if (value != 0):
#                             nonzerocount += 1
#                         left_hemi[t][x][y][z] = value
#
#         log_file.write("l_hemi has shape " + str(left_hemi.shape)+"\n\n")
#
#
#         left_encoded_responses.append(left_hemi)
#
#         print("slicing right hemi of response " + str(response) + "...")
#         r_hemi = numpy.zeros((TIMESTEPS, r_x, r_y, r_z))
#
#         for t in range(0, TIMESTEPS):
#             for x in range(0, r_x):
#                 for y in range(0, r_y):
#                     for z in range(0, r_z):
#                         r_hemi[t][x][y][z] = volume[t][r_x_min + x][r_y_min + y][r_z_min + z]
#
#         right_encoded_responses.append(r_hemi)
#
#         return [l_x, l_x_min, l_y, l_y_min, l_z, l_z_min, r_x, r_x_min, r_y, r_y_min, r_z, r_z_min]

def get_voxel_bounds(voxels, arg_dict, log_file):
    subject = arg_dict["subject"]
    roi = arg_dict["ROI"]


    min_x = -1
    max_x = -1
    l_x_min = -1
    l_x_max = -1
    r_x_min = -1
    r_x_max = -1
    x_dict = {}
    #### figure out the spread of x values
    for voxel in voxels:
        if min_x == -1:
            min_x = voxel[0]
        elif voxel[0] < min_x:
            min_x = voxel[0]
        elif voxel[0] > max_x:
            max_x = voxel[0]

    l_x_min = min_x
    r_x_max = max_x
    for m in range(0, max_x+1):
        x_dict[m] = 0
    #Flag all the x values with active voxels
    for voxel in voxels:
        x_val = voxel[0]
        x_dict[x_val] += 1

    for m in range(min_x, max_x+1):
        #if that's an active voxel
        if x_dict[m]!=0:
            #if left hemisphere's max was already found and right hemisphere's min has not been found, then this is r_x_min
            if l_x_max !=- 1 and r_x_min == -1:
                r_x_min = m

        if x_dict[m]==0:
            # if we find a zero and haven't found l_x_max, then the previous one must be l_x_max
            if l_x_max == -1:
                l_x_max = m-1

    ##############
    ##############################
    ################################################
    #the above code doesn't work on contiguous regions of the brain because there's no gap
    #let's try loading the two regions directly
    # ds = get_subject_ds(subject)
    # rois = [int(roi)]
    # left_ds_masked = mask_subject_ds(ds, subject, rois)
    # left_min_x = -1
    # left_max_x = -1
    # log_file.write("left_ds_masked voxel indices are "+str(left_ds_masked.fa.voxel_indices)+"\n")
    # for voxel in left_ds_masked.fa.voxel_indices:
    #     if left_min_x == -1:
    #         left_min_x = voxel[0]
    #     elif voxel[0] < left_min_x:
    #         left_min_x = voxel[0]
    #     elif voxel[0] > left_max_x:
    #         left_max_x = voxel[0]
    #
    # rois = [int(roi)+1000]
    # right_ds_masked = mask_subject_ds(ds, subject, rois)
    # right_min_x = -1
    # right_max_x = -1
    # log_file.write("right_ds_masked voxel indices are "+str(right_ds_masked.fa.voxel_indices)+"\n")
    #
    # for voxel in right_ds_masked.fa.voxel_indices:
    #     if right_min_x == -1:
    #         right_min_x = voxel[0]
    #     elif voxel[0] < right_min_x:
    #         right_min_x = voxel[0]
    #     elif voxel[0] > right_max_x:
    #         right_max_x = voxel[0]
    #
    # l_x_min = left_min_x
    # l_x_max = left_max_x
    # r_x_min = right_min_x
    # r_x_max = right_max_x
##########################################################

    log_file.write("l_x_min and l_x_max are "+str(l_x_min)+","+str(l_x_max))
    log_file.write("r_x_min and r_x_max are "+str(r_x_min)+","+str(r_x_max))

    ##### so we've found the x value ranges for the two hemispheres.  Now it's easy to find the ranges on the other two dimensions
    l_y_min = -1
    l_y_max = -1
    l_z_min = -1
    l_z_max = -1

    r_y_min = -1
    r_y_max = -1
    r_z_min = -1
    r_z_max = -1

    for voxel in voxels:
        i = voxel[0]
        j = voxel[1]
        k = voxel[2]

        # if it's in the left side
        if i <= l_x_max:
            if l_y_min == -1:
                l_y_min = j
            elif j < l_y_min:
                l_y_min = j
            elif j > l_y_max:
                l_y_max = j

            if l_z_min == -1:
                l_z_min = k
            elif k < l_z_min:
                l_z_min = k
            elif k > l_z_max:
                l_z_max = k

        #equivalent to an else statement, but for clarity...
        elif i >= r_x_min:
            if r_y_min == -1:
                r_y_min = j
            elif j < r_y_min:
                r_y_min = j
            elif j > r_y_max:
                r_y_max = j

            if r_z_min == -1:
                r_z_min = k
            elif k < r_z_min:
                r_z_min = k
            elif k > r_z_max:
                r_z_max = k

    l_x = l_x_max - l_x_min + 1
    l_y = l_y_max - l_y_min + 1
    l_z = l_z_max - l_z_min + 1
    r_x = r_x_max - r_x_min + 1
    r_y = r_y_max - r_y_min + 1
    r_z = r_z_max - r_z_min + 1

    log_file.write("x values range from "+str(l_x_min)+" to "+str(l_x_max)+" and "+str(r_x_min)+" to "+str(r_x_max)+"\n")
    log_file.write("y values range from "+str(l_y_min)+" to "+str(l_y_max)+" and "+str(r_y_min)+" to "+str(r_y_max)+"\n")
    log_file.write("z values range from "+str(l_z_min)+" to "+str(l_z_max)+" and "+str(r_z_min)+" to "+str(r_z_max)+"\n")

    return l_x, l_x_min, l_y, l_y_min, l_z, l_z_min, r_x, r_x_min, r_y, r_y_min, r_z, r_z_min

#Returns python lists of the left and right hemispheres after max pooling
#The lists are flat by construction, they are simply appended to as we move through the 3D space looking for pooled values.
#Applies tanh to the pooled value when it appends it to the list
#Returns left side (lower x-indices, which is right brain) and right side as two separate lists
def pooling(orig_data, responses, voxel_indices, num_filters, spatial_window, pooling_cube, log_file, arg_dict):
    l_x, l_x_min, l_y, l_y_min, l_z, l_z_min, r_x, r_x_min, r_y, r_y_min, r_z, r_z_min = get_voxel_bounds(voxel_indices, arg_dict, log_file)
    l_x_max = l_x_min + l_x
    r_x_max = r_x_min + r_x
    l_y_max = l_y_min + l_y
    r_y_max = r_y_min + r_y
    l_z_max = l_z_min + l_z
    r_z_max = r_z_min + r_z

    pool_count = 0
    left_count = -1

    left_result = []
    right_result = []


    pooled_3d_r = []
    pooled_3d_l = []
    TIMESTEPS = arg_dict["TIMESTEPS"]
    # for use when we're not splitting the brain and only using one layer in the overal architecture
    flat_pooled = []


    for filter in range(0, num_filters):

        cube_x = int(pooling_cube[0])
        cube_y = int(pooling_cube[1])
        cube_z = int(pooling_cube[2])
        #print("pooling for filter "+str(filter)+"...\n")
        l_append_count = 0

        timeslices_l = []
        timeslices_r = []

        flat_timeslices = []

        #fill the 3d space with encoded values for this filter
        for t in range(0, TIMESTEPS):
            pooled_count = 0

            #if(t%100==0):
                #print("pooling for timestep "+str(t)+"...\n")
            count = 0
            for voxel in voxel_indices:
                x = voxel[0]
                y = voxel[1]
                z = voxel[2]
                #print("t, x, y, z: "+str(t)+","+str(x)+","+str(y)+","+str(z))
                #print("filter, t, count: "+str(filter)+","+str(t)+","+str(count))
                orig_data[t][x][y][z] = responses[filter][t][count]
                count += 1
            # if(t==1 and filter==1):
            #     tzero3d=orig_data[0]
            #     tzero3d_good = tzero3d[25:43]
            #
            #     #log_file.write("tzero3d left: \n ------------\n")
            #     logcount=0
            #     logzerocount=0
            #     for xd in range(25, 43):
            #         for yd in range(63, 69):
            #             for zd in range(17, 28):
            #                 log_file.write(str(tzero3d[xd][yd][zd]))
            #                 if(tzero3d[xd][yd][zd]==0):
            #                     logzerocount+=1
            #                 else:
            #                     logcount+=1
            #             log_file.write("\n")
            #         log_file.write("\n -------------- \n")

                #log_file.write("\n\n\n")

                #log_file.write("tzero3d right: \n ------------\n")
                # for xd in range(33, 42):
                #     for yd in range(63, 69):
                #         for zd in range(17, 27):
                #             #log_file.write(str(tzero3d[xd][yd][zd]))
                #             if(tzero3d[xd][yd][zd]==0):
                #                 logzerocount+=1
                #             else:
                #                 logcount+=1
                        #log_file.write("\n")
                #log_file.write("\n -------------- \n")


                #log_file.write("\n\n\n")
                #log_file.write("logzerocount is "+str(logzerocount)+"\n")
                #log_file.write("logcount is "+str(logcount)+"\n\n")

            #pool over the whole space, storing in appropriate lists
            x_slices_l = []
            x_slices_r = []

            flat_slice = []

            x_min = l_x_min
            x_max = r_x_max

            y_min = min(l_y_min, r_y_min)
            if (l_y_max > r_y_max):
                y_max = l_y_max
            else:
                y_max = r_y_max

            z_min = min(l_z_min, r_z_min)
            if (l_z_max > r_z_max):
                z_max = l_z_max
            else:
                z_max = r_z_max

            #variables to help us build the 3D version of the pooled values
            #currently unused
            z_height = 0
            temp_z = 0
            y_height = 0
            temp_y = 0
            x_height = 0
            temp_x = 0


            # max pool if the area is mostly non zero
            for x in range(x_min, x_max + 1, cube_x):
                y_slice = []
                if x > l_x_max and left_count == -1:
                    left_count = pool_count
                for y in range(y_min, y_max+1, cube_y):
                    z_slice = []
                    for z in range(z_min, z_max+1, cube_z):
                        cube = get_cube(orig_data, t, x, y, z, cube_x, cube_y, cube_z)
                        # if (t == 0 & filter == 0):
                        #     log_file.write("checking cube "+str(cube)+"\n")
                        result = zero_check(cube, cube_x, cube_y, cube_z, "primary")
                        #if (t == 0 & filter == 0):
                            #log_file.write("got result "+str(result)+"\n")
                        if (len(result)!=0):
                            max = numpy.amax(result)
                            pooled_count += 1
                            #build the list of pooled values in the z direction
                            # ENABLE FOR TWO LAYER FUNCTIONALITY TO PRESERVE 3D STRUCTURE
                            #z_slice.append(tanh(max))
                            flat_slice.append(tanh(max))
                            pool_count += 1

                    # ENABLE FOR TWO LAYER FUNCTIONALITY
                    #if(len(z_slice)>0):
                        #y_slice.append(z_slice)
                    if z_height < len(z_slice):
                        z_height = len(z_slice)

                # ENABLE FOR TWO LAYER FUNCTIONALITY
                #if x <= l_x_max and len(y_slice) > 0:
                    #x_slices_l.append(y_slice)
                #elif x>=r_x_min and len(y_slice) > 0:

                    #ENABLE FOR SPLIT BRAIN FUNCTIONALITY
                    #x_slices_r.append(y_slice)
                    #x_slices_l.append(y_slice)


                if y_height < len(y_slice):
                    y_height = len(y_slice)
            #print("end of timestep, x_slices_r is "+str(x_slices_r))
#             print("x_slices_r has length "+str(len(x_slices_r)))
#             print("its zeroth index has length "+str(len(x_slices_r[0])))
#             print("and the zeroth index of that has length "+str(len(x_slices_r[0][0])))
            #Add the pooled data for this timeslice
            # if(t==0):
            #     log_file.write("x_slices_l has length "+str(len(x_slices_l)))
            #     log_file.write("its first thing has length "+str(len(x_slices_l[0])))
            #     log_file.write("the first thing is "+str((x_slices_l[0])))
            if(t==0 & filter==0):
                log_file.write("pooled count is "+str(pooled_count)+"\n\n")

            # ENABLE FOR TWO LAYER FUNCTIONALITY
            #timeslices_l.append(x_slices_l)

            #l_append_count += 1
            # ENABLE FOR SPLIT BRAIN FUNCTIONALITY
            #timeslices_r.append(x_slices_r)

            flat_timeslices.append(flat_slice)

        #print("we appended "+str(l_append_count)+" timeslices on filter "+str(filter)+"\n")
        #print("the length of timeslices_l is "+str(len(timeslices_l)))

        # ENABLE FOR TWO LAYER FUNCTIONALITY
        #pooled_3d_l.append(timeslices_l)
        # ENABLE FOR SPLIT BRAIN FUNCTIONALITY
        #pooled_3d_r.append(timeslices_r)

        flat_pooled.append(flat_timeslices)


    #returned tensors are num_filters x timesteps x X x Y x Z

    # ENABLE FOR SPLIT BRAIN/TWO LAYER FUNCTIONALITY
    #return pooled_3d_l, pooled_3d_r

    log_file.write("----------------------")
    log_file.write("flat_pooled has length "+str(len(flat_pooled))+"\n")
    log_file.write("left count is "+str(left_count)+"\n")
    log_file.write("each of those has length "+str(len(flat_pooled[0]))+"\n")
    log_file.write("each of THOSE has length "+str(len(flat_pooled[0][0]))+"\n\n")
    log_file.write("----------------------")

    return flat_pooled, left_count

#Checks of the desired cube should be used for max pooling or not.
#We really only need to identify these cubes once, as they are the same for each time slice. Something for future version.
def zero_check(cube, c_x, c_y, c_z, layer):
    values = []
    if layer=="primary":
        ZERO_THRESHOLD = ZERO_THRESHOLD_PRIMARY
    elif layer=="secondary":
        ZERO_THRESHOLD = ZERO_THRESHOLD_SECONDARY
    count = 0
    for x in range(0,c_x):
        for y in range(0,c_y):
            for z in range(0,c_z):
                if (cube[x][y][z] != 0):
                    values.append(cube[x][y][z])
                else:
                    #the number of zeroes
                    count += 1
    #ZERO_THRESHOLD defined in Constants.py
    if count/float(c_x*c_y*c_z) > ZERO_THRESHOLD:
        return []
    else:
        return values


#Build a numpy array of the 3D sub-cube in question.
def get_cube(data, t, x, y, z, c_x, c_y, c_z):
    if(t!=-1):
        data = data[t]
    x_len = len(data)
    y_len = len(data[0])
    z_len = len(data[0][0])
    if (z + c_z > z_len):
        c_z = z_len - z
    if (y + c_y > y_len):
        c_y = y_len - y
    if (x + c_x > x_len):
        c_x = x_len - x
    if (c_x == 0 or c_y == 0 or c_z == 0):
        print("a cube dimension was zero for x,y,z of " + str(x) + "," + str(y) + "," + str(z))
    cube = numpy.zeros((c_x, c_y, c_z))
    for i in range(0, c_x):
        for j in range(0, c_y):
            for k in range(0, c_z):
                cube[i][j][k] = data[x + i][y + j][z + k]

    return cube

def flatten_3d_responses(left, right, arg_dict, log_file):
    flat = []
    num_primary_filters = arg_dict["num_primary_filters"]
    #print("in flatten 3d responses, left has length "+str(len(left))+"\n")
    #print("left 00 has length "+str(len(left[0][0]))+" which is "+str(left[0][0])+"\n")
    for i in range(0, num_primary_filters):
        concat, l_count, r_count = flatten_3d_timeseries(left[i], right[i], log_file)
        flat.append(concat)

    return flat, l_count, r_count

#flattens TIMESTEPS x CUBE into TIMESTEPS x SAMPLES
def flatten_3d_timeseries(left, right, log_file):
    TIMESTEPS = arg_dict["TIMESTEPS"]
    new_left = []
    new_right = []
    #print("in flatten 3d timeseries, left has length "+str(len(left)))
    for t in range(0, TIMESTEPS):
        #Replace the 3d list at time t with its flattened version
        timeslice = left[t]
        temp = [item for items in timeslice for item in items]

        flatslice = [item for items in temp for item in items]

        new_left.append(flatslice)
        lcount = len(flatslice)

    print("lcount is " + str(lcount))

    for t in range(0, TIMESTEPS):
        # Replace the 3d list at time t with its flattened version
        timeslice = right[t]
        temp = [item for items in timeslice for item in items]

        flatslice = [item for items in temp for item in items]

        new_right.append(flatslice)
        rcount = len(flatslice)

    #Concatenate for each timestep
    for t in range(0, TIMESTEPS):
        new_left[t] = new_left[t] + new_right[t]
    log_file.write("length of newleft is "+str(len(new_left))+"\n")

    return new_left, lcount, rcount

#Takes a 3d python list and returns the dimensions of the smallest (padded) cube that can contain it, as the list itself is not a perfect cube
def get_padded_dimensions(list_3d, log_file):
    cube = list_3d[0][0]

    x_dim = len(cube)
    log_file.write("x_dim is "+str(x_dim)+"\n")
    y_dim = -1
    z_dim = -1
    y_lengths = []
    z_lengths = []

    for y in range(0, x_dim):
        #if we go further in the y direction here then before, update the size of the y_dimension
        if (len(cube[y]) > y_dim):
            y_dim = len(cube[y])
        y_lengths.append(len(cube[y]))
        for z in range(0, len(cube[y])):
            if(len(cube[y][z]) > z_dim):
                z_dim = len(cube[y][z])
            z_lengths.append(len(cube[y][z]))
    return [x_dim, y_dim, z_dim, y_lengths, z_lengths]

#gets samples fresh from the loaded dataset object
#strategy is a string telling us which strategy is being used
def do_strategy(arg_dict, samples, log_file):
    my_samples = np.array(samples)
    strat = arg_dict["strategy"]
    TIMESTEPS = arg_dict["TIMESTEPS"]
    label_range=range(152,190)+range(252,290)+range(352,390)+range(452,490)

    #Remove all labelled samples
    if strat=="1":
        if(arg_dict["half"]==0):
            start_run=1
            end_run=8
        elif(arg_dict["half"]==1):
            start_run=1
            end_run=4
        elif(arg_dict["half"]==2):
            start_run=5
            end_run=8
        new_steps = []
        step=0
        strat_samples=[]
        labelled_steps=[]

        accession=ACCESSIONS[arg_dict["subject"]]

        for run in range(start_run,end_run+1):
            #access Accession file
            filepath="/isi/music/auditoryimagery2/targets/"+str(accession)+"_run-0"+str(run)+".txt"
            with open(filepath) as fp:
                line=fp.readline()
                while line:
                    if int(line.strip()) in label_range:
                        log_file.write("labelled timestep at t="+str(step)+", the label was "+str(line.strip())+"\n")
                        labelled_steps.append(step)
                    else:
                        new_steps.append(step)
                    step+=1
                    line=fp.readline()

        for i in range(0, len(new_steps)):
            t=new_steps[i]
            strat_samples.append(samples[t])
        strat_steps = len(strat_samples)
        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat steps is "+str(strat_steps))
        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat samples is "+str(strat_samples))

    #Create dataset filtered by second half, i.e hold out first half during training
    elif strat=="2":
        # access Accession file
        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = []
        accession=ACCESSIONS[arg_dict["subject"]]


        # count through the first four runs
        for run in range(1, 5):
            # access Accession file
            filepath="/isi/music/auditoryimagery2/targets/"+str(accession)+"_run-0"+str(run)+".txt"
            with open(filepath) as fp:
                line=fp.readline()
                while line:
                    step+=1
                    line=fp.readline()

        for t in range(step,1864):
            strat_samples.append(samples[t])

        strat_steps = len(strat_samples)

        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat steps is "+str(strat_steps))
        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat samples is "+str(strat_samples))

    #Create dataset filtered by first half, i.e hold out second half during training
    elif strat=="3":
        # access Accession file
        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = []
        accession=ACCESSIONS[arg_dict["subject"]]


        #count through the first four runs
        for run in range(1, 5):
            # access Accession file
            filepath="/isi/music/auditoryimagery2/targets/"+str(accession)+"_run-0"+str(run)+".txt"
            with open(filepath) as fp:
                line=fp.readline()
                while line:
                    step+=1
                    line=fp.readline()

        #step is now the timestep after the first four runs
        for t in range(0,step):
            strat_samples.append(samples[t])

        strat_steps = len(strat_samples)

        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat steps is "+str(strat_steps))
        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat samples is "+str(strat_samples))

    #Create dataset filtered by all unlabeled data, and second half of labeled data
        # i.e hold out first half of labeled data
    elif strat=="4":
    #access Accession file
        step = 0
        new_steps=[]
        strat_samples=[]
        labelled_steps=[]

        accession = ACCESSIONS[arg_dict["subject"]]

        for run in range(1,5):
            #access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    if int(line.strip()) in label_range:

                        log_file.write("labelled timestep at t="+str(step)+", the label was "+str(line.strip())+"\n")
                        labelled_steps.append(step)

                    else:
                        new_steps.append(step)
                    step+=1
                    line = fp.readline()

        #add the samples at the unlabeled timesteps in the first half
        for i in range(0,len(new_steps)):
            t=new_steps[i]
            strat_samples.append(samples[t])

        #add the samples from all timesteps in the second half
        for i in range(step,1864):
            strat_samples.append(samples[i])

        strat_steps = len(strat_samples)

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))

    #Create dataset filtered by all unlabeled data, and first half of labeled data
        # i.e hold out second half of labeled data
    elif strat=="5":
        # access Accession file
        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = []

        accession = ACCESSIONS[arg_dict["subject"]]

        # add all the samples in the first half, as well as count them with "step"
        for run in range(1, 5):
            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    new_steps.append(step)
                    step += 1

        #remove all labelled samples in the second half
        for run in range(5,9):
            #access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    if int(line.strip()) in label_range:
                        log_file.write("labelled timestep at t="+str(step)+", the label was "+str(line.strip())+"\n")
                        labelled_steps.append(step)
                    else:
                        new_steps.append(step)
                    step+=1


        #new_steps contains both halves, so add them all
        for i in range(0, len(new_steps)):
            t = new_steps[i]
            strat_samples.append(samples[t])

        strat_steps=len(strat_samples)

        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat steps is "+str(strat_steps))
        log_file.write("for strategy "+str(arg_dict["strategy"])+", strat samples is "+str(strat_samples))


    else:
        log_file.write("error in do strategy, illegal strategy")



    return strat_samples, strat_steps, labelled_steps

def make_StratDS(arg_dict, log_file):
    strat=arg_dict["strategy"]
    half = arg_dict["half"]
    ROI=arg_dict["ROI"]
    subject=arg_dict["subject"]
    TIMESTEPS = arg_dict["TIMESTEPS"]
    #condition labels for HT, HC, IT, IC (maybe not in that order)
    label_range=range(152,190)+range(252,290)+range(352,390)+range(452,490)

    ds = load_subject(subject, ROI, log_file)
    samples = ds.samples

    if strat == "0":
        strat_samples = samples[:]
        strat_steps = 932
        # arg_dict["strat_steps"]=strat_steps #flat number of timesteps
        if arg_dict["half"] == 1:
            new_steps = list(range(0, 932))
        elif arg_dict["half"]==2:
            new_steps = list(range(932, 1084))
        # arg_dict["new_steps"]=new_steps #list of timesteps
        # arg_dict["strat_targets"]=strat_targets
        #arg_dict["strat_targets"] = None
        # arg_dict["labelled_steps"]=labelled_steps
        labelled_steps = None

    # Remove all labelled samples
    elif strat == "1":
        if(arg_dict["half"]==0):
            start_run=1
            end_run=8
        elif(arg_dict["half"]==1):
            start_run=1
            end_run=4
        elif(arg_dict["half"]==2):
            start_run=5
            end_run=8
        new_steps = []
        step = 0
        strat_samples = []
        labelled_steps = [] #tells us which steps we can't overlap with when creating "winow samples" during AE training.

        accession = ACCESSIONS[arg_dict["subject"]]

        for run in range(start_run, end_run):
            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"
            with open(filepath) as fp:
                line = fp.readline()
                while line:
                    if int(line.strip()) in label_range:
                        log_file.write(
                            "labelled timestep at t=" + str(step) + ", the label was " + str(line.strip()) + "\n")
                        labelled_steps.append(step)
                    else:
                        new_steps.append(step)
                    step += 1
                    line = fp.readline()

        for i in range(0, len(new_steps)):
            t = new_steps[i]
            strat_samples.append(samples[t])

        strat_samples=np.array(strat_samples)
        strat_steps = len(strat_samples)

        if(strat_steps!=len(new_steps)):
            log_file.write("step length mismatch for strat "+str(strat)+": strat steps is "+str(strat_steps)+" and new steps has length "+str(len(new_steps))+"\n")

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))


    # Create dataset filtered by second half, i.e hold out first half during training

    elif strat == "2":

        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = [] #the AE cannot possibly overlap its window samples with test data with this strategy, so it remains empty

        accession = ACCESSIONS[arg_dict["subject"]]

        # count through the first four runs

        for run in range(1, 5):

            # access Accession file

            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"

            with open(filepath) as fp:

                line = fp.readline()

                while line:
                    step += 1

                    line = fp.readline()

        for t in range(step, 1864):
            new_steps.append(step)
            strat_samples.append(samples[t])

        strat_samples=np.array(strat_samples)
        strat_steps = len(strat_samples)

        if (strat_steps != len(new_steps)):
            log_file.write("step length mismatch for strat " + str(strat) + ": strat steps is " + str(
                strat_steps) + " and new steps has length " + str(len(new_steps)) + "\n")

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))


    # Create dataset filtered by first half, i.e hold out second half during training

    elif strat == "3":

        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = [] #the AE cannot possibly overlap its window samples with test data with this strategy, so it remains empty
        accession = ACCESSIONS[arg_dict["subject"]]

        # count through the first four runs
        for run in range(1, 5):
            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"

            with open(filepath) as fp:
                line = fp.readline()

                while line:
                    step += 1
                    line = fp.readline()

        # step is now the timestep after the first four runs
        for t in range(0, step):
            strat_samples.append(samples[t])
            new_steps.append(t)

        strat_steps = len(strat_samples)
        strat_samples=np.array(strat_samples)

        if (strat_steps != len(new_steps)):
            log_file.write("step length mismatch for strat " + str(strat) + ": strat steps is " + str(
                strat_steps) + " and new steps has length " + str(len(new_steps)) + "\n")

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))


    # Create dataset filtered by all unlabeled data, and second half of labeled data

    # i.e hold out first half of labeled data

    elif strat == "4":

        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = [] #in this case, the labelled steps in the first half, we dont want "window samples" to overlap with these
        accession = ACCESSIONS[arg_dict["subject"]]



        for run in range(1, 5):

            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"

            with open(filepath) as fp:

                line = fp.readline()
                while line:

                    if int(line.strip()) in label_range:
                        log_file.write("labelled timestep at t=" + str(step) + ", the label was " + str(line.strip()) + "\n")
                        labelled_steps.append(step)

                    else:
                        new_steps.append(step)

                    step += 1
                    line = fp.readline()

        # add the samples at the unlabeled timesteps in the first half
        for i in range(0, len(new_steps)):
            t = new_steps[i]

            strat_samples.append(samples[t])

        # add the samples from all timesteps in the second half

        for i in range(step, 1864):
            strat_samples.append(samples[i])
            new_steps.append(i)

        strat_steps = len(strat_samples)
        strat_samples=np.array(strat_samples)

        if (strat_steps != len(new_steps)):
            log_file.write("step length mismatch for strat " + str(strat) + ": strat steps is " + str(
                strat_steps) + " and new steps has length " + str(len(new_steps)) + "\n")

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))


    # Create dataset filtered by all unlabeled data, and first half of labeled data

    # i.e hold out second half of labeled data

    elif strat == "5":

        step = 0
        new_steps = []
        strat_samples = []
        labelled_steps = []

        accession = ACCESSIONS[arg_dict["subject"]]

        # add all the samples in the first half, as well as count them with "step"
        for run in range(1, 5):

            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"

            with open(filepath) as fp:

                line = fp.readline()
                while line:
                    new_steps.append(step)

                    step += 1

        # remove all labelled samples in the second half

        for run in range(5, 9):

            # access Accession file
            filepath = "/isi/music/auditoryimagery2/targets/" + str(accession) + "_run-0" + str(run) + ".txt"

            with open(filepath) as fp:
                line = fp.readline()

                while line:
                    if int(line.strip()) in label_range:
                        log_file.write(
                            "labelled timestep at t=" + str(step) + ", the label was " + str(line.strip()) + "\n")
                        labelled_steps.append(step)

                    else:
                        new_steps.append(step)

                    step += 1

        # new_steps contains both halves, so add them all
        for i in range(0, len(new_steps)):
            t = new_steps[i]

            strat_samples.append(samples[t])

        strat_steps = len(strat_samples)
        strat_samples = np.array(strat_samples)

        if (strat_steps != len(new_steps)):
            log_file.write("step length mismatch for strat " + str(strat) + ": strat steps is " + str(
                strat_steps) + " and new steps has length " + str(len(new_steps)) + "\n")

        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat steps is " + str(strat_steps))
        log_file.write("for strategy " + str(arg_dict["strategy"]) + ", strat samples is " + str(strat_samples))

    else:
        log_file.write("Illegal strat value, big error\n\n")


    #at this point, strat_samples has the correct samples, labelled_steps tells us where we can't overlap while making window_samples later,
        # new_steps is the list of training data timesteps, and strat_steps is the flat number of timesteps using in training

    #what remains is to create the targets array and the 3D tensor for the timesteps indicated by new_steps

    strat_targets = []
    targets=ds.targets

    for t in new_steps:
        strat_targets.append(targets[t])

    return strat_samples, new_steps, strat_targets, strat_steps, labelled_steps



