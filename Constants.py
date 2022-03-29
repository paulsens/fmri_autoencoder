#SUBJECT IDs:
SUBJECTS = ['sid001401', 'sid001415', 'sid000388', 'sid001419', 'sid001410', 'sid001541',
             'sid001427', 'sid001088', 'sid001564', 'sid001581', 'sid001571', 'sid001594',
             'sid001660', 'sid001661', 'sid001664', 'sid001665', 'sid001125', 'sid001668',
             'sid001672', 'sid001678', 'sid001680']

#See paper regarding excluded subjects
RED_FLAGGED = ["sid001415", "sid000388", "sid001564", "sid001594", "sid001613", "sid001679"]

YELLOW_FLAGGED = ["sid001125", "sid001660"]

#WHETHER TO SWAP RUNS SO THAT ALL SUBJECTS HAVE THE SAME ORDER
SWAP_F = True
SWAP_RUNS = [2, 1, 4, 3, 6, 5, 8, 7]

TIMESTEPS = 1864

#Path to directory where you want subject ROIs to be saved. A folder for each subject will be created at this location.
#Do not include final slash
ROI_PATH = "set_me"

#PATH TO PYTHON CODE DIRECTORY
#Do not include final slash
CODE_PATH = "set_me"

#PATH TO FOLDER CONTAINING SUBJECT FOLDERS DOWNLOADED FROM BRAINLIFE
#Do not include final slash
DATA_PATH = "set_me"

#PATH TO FOLDER CONTAINING EXTRACTED "EXTRAS" FROM GITHUB REPO
#Do not include final slash
EXTRAS_PATH = "set_me"

#PATH TO FOLDER CONTAINING TXT LOGS THAT RUN DURING the main pipeline
#Do not include final slash
LOGS_PATH = "set_me"

ACCESSIONS = {'sub-sid001401':'A002636',
'sub-sid001415':	'A002652',
'sub-sid000388':	'A002655',
'sub-sid001419':	'A002659',
'sub-sid001410':	'A002677',
'sub-sid001541':	'A002979',
'sub-sid001427':	'A002996',
'sub-sid001088':	'A003000',
'sub-sid001564':	'A003037',
'sub-sid001581':	'A003067',
'sub-sid001594':	'A003098',
'sub-sid001571':	'A003219',
'sub-sid001660':	'A003231',
'sub-sid001613':	'A003232',
'sub-sid001661':	'A003233',
'sub-sid001664':	'A003238',
'sub-sid001665':	'A003239',
'sub-sid001125':	'A003243',
'sub-sid001668':	'A003247',
'sub-sid001672':	'A003259',
'sub-sid001678':	'A003271',
'sub-sid001679':	'A003272',
'sub-sid001680':	'A003274'}

MRISPACE= 'MNI152NLin2009cAsym' # 'MNI152NLin2009cAsym #CHANGE THIS
PARCELLATION='desc-aparcaseg_dseg'

#CONVOLUTION CONSTANTS
TEMPORAL_WINDOW1 = 6 #takes 12 seconds for human neuron activity to start and stop, each slice is 2 seconds
SPATIAL_WINDOW1 = 16 #the paper does not say what this value was, "retract the number of dimensions comparable with the standard MVPA method"
NUM_PRIMARY_FILTERS=14
POOLING_CUBE = [3,3,3] #Dimensions of sliding cube used for pooling (with no overlaps)
ZERO_THRESHOLD_PRIMARY = 0.6

#PRIMARY AE CONSTANTS
P_BATCH_SIZE = 32
P_KERNEL_REGULARIZER = 0.001
P_ACTIVITY_REGULARIZER = 0.001
P_EPOCHS = 30

#What is this for?
left = -1
right = -1
