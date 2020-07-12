from Model.CNNSeq2Seq import data_preprocess as dp

# Just for not producing error Need to Remove SRC & TRG with corr. value during Data Pre-Processing:

INPUT_DIM = len(dp.SRC.vocab)
OUTPUT_DIM = len(dp.TRG.vocab)
EMB_DIM = 256
HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10  # number of conv. blocks / layers in encoder
DEC_LAYERS = 10  # number of conv. blocks / layers in decoder
ENC_KERNEL_SIZE = 3  # must be odd!
DEC_KERNEL_SIZE = 3  # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
BATCH_SIZE = 128

# For Training Gradients
CLIP = 0.1

MAX_LENGTH = 61

# Model Train Parameters
EPOCH = 25
Learning_Rate = 0.001

# Setting up special Tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# Vector value of Special Tokens
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2
PAD_TOKEN_IDX = 0
UNK_TOKEN_IDX = 3
