SAMPLING_RATE = 16_000

# Log Mel Config
N_FFT=400
N_MELS=64
HOP_LEN=int(0.010 * SAMPLING_RATE)  # 10ms - 160 samples 
WIN_LEN=int(0.025 * SAMPLING_RATE)  # 25ms - 400 samples

# Spec Augment
TW_P, TW_LIM = 0.02, 50         # upto 0.5s of distortion
TM1_P, TM1_LIM = 0.01, 20       # upto 200ms of masking at an instant
TM2_P, TM2_LIM = 0.01, 2        # upto 20ms of masking at an instant
FM_P, FM_LIM = 0.01, 10         # upto 10 bands of frequencies

TW_SCL_MIN, TW_SCL_RANGE = 1.25, 0.75       # [1.25, 2.00]
TW_PIV_MIN, TW_PIV_RANGE = 1.50, 1.00       # [1.50, 2.50]

# Text
N_VOCAB = 5_000                # i think going shorter than usual is good because ur getting really micro level inputs and u probably wont be getting whole ass words like that
DISCARDED_SYMS = ":;,.'\"?-!؟:’۔‘،"

# CTC Network
CTC_IN_SIZE = N_MELS
CTC_H_SIZE = 512
CTC_N_LAYERS = 3
CTC_OUT_SIZE = N_VOCAB + 1
CTC_DROPOUT = 0.1

BLANK_IDX = N_VOCAB

# Conformer

CNF_D_MODEL = 256
CNF_N_BLOCKS = 16
CNF_OUT_SIZE = 512
CNF_CHUNK_MAX_LEN = 2**10

SUBSAMPLER_N_FILTERS = 32
SUBSAMPLER_KERNEL_SIZE = 3
SUBSAMPLER_STRIDE = 2
SUBSAMPLER_PAD = 1

CNF_MHSA_D_HEAD = 128
CNF_MHSA_N_HEADS = 8

CNF_CONV_N_FILTERS = 2 * CNF_D_MODEL
CNF_CONV_KERNEL_SIZE = 31
CNF_CONV_STRIDE = 1
CNF_CONV_PAD = 15

SUBSAMPLER_PROJ_DROPOUT = 0.1
CNF_FFNN_DROPOUT = 0.1
CNF_MHSA_DROPOUT = 0.1
CNF_CONV_DROPOUT = 0.1

# Everything Else in the 3-way joint Conformer-Transducer

PRED_IN_RANGE = N_VOCAB   # never gets a blank as input
PRED_EMBEDDING_SIZE = 512
PRED_H_SIZE = 256
PRED_N_LAYERS = 3
PRED_OUT_SIZE = 512
PRED_DROPOUT = 0
PRED_BIDIRECTIONAL = False

LINK_IN_SIZE = N_VOCAB + 1
LINK_H_SIZE = 256
LINK_N_LAYERS = 1
LINK_OUT_SIZE = 512
LINK_DROPOUT = 0
LINK_BIDIRECTIONAL = True

JOINT_IN_SIZE = CNF_OUT_SIZE + PRED_OUT_SIZE + LINK_OUT_SIZE
JOINT_H_SIZE = 128
JOINT_OUT_SIZE= N_VOCAB + 1