import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import random
import math
import time

# Import Data Pre-process Files, Model, and Config File
from Model.EngHindiDataPreprocess import config
from Model.CNNSeq2Seq import model
from Model.EngHindiDataPreprocess.eng_hin_vocab_creator import ENG_DATA_PADDED, HIN_DATA_PADDED

# Import Train-Test Splitter
from sklearn.model_selection import train_test_split

# Manually Cleaning / Deleting Unused Tensors
# import gc

import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

# Setting Parameters
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Set Batch Size
BATCH_SIZE = config.BATCH_SIZE

# Convert Integer Dataset to Tensor
tensor_eng = torch.tensor(ENG_DATA_PADDED, dtype=torch.int32)
tensor_hin = torch.tensor(HIN_DATA_PADDED, dtype=torch.int32)

# Split Dataset to Train and Validation Set
X_train, X_val, y_train, y_val = train_test_split(tensor_eng, tensor_hin, test_size=0.1)

# Convert the Split Dataset to Dataset Type
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_val, y_val)

# Make Loader or Batchify the Split Datasets
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, drop_last=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, drop_last=True, num_workers=0)

# exit(0)

# Setting up Model Parameters and Initializing the Model
INPUT_DIM = config.INPUT_DIM
OUTPUT_DIM = config.OUTPUT_DIM
EMB_DIM = config.EMB_DIM
HID_DIM = config.HID_DIM  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 12  # number of conv. blocks in encoder
DEC_LAYERS = 12  # number of conv. blocks in decoder
ENC_KERNEL_SIZE = config.ENC_KERNEL_SIZE  # must be odd!
DEC_KERNEL_SIZE = config.DEC_KERNEL_SIZE  # can be even or odd
ENC_DROPOUT = config.ENC_DROPOUT
DEC_DROPOUT = config.DEC_DROPOUT
TRG_PAD_IDX = config.PAD_TOKEN_IDX

enc = model.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = model.Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = model.CNNSeq2Seq(enc, dec).to(device)


# For checking num. of trainable parameters in Model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model):,} trainable parameters')

# Setting up Optimizer and Loss function for Training
optimizer = optim.Adam(model.parameters(), lr=config.Learning_Rate)
# scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Comment Out exit(0) Only in case to Train the model
# exit(0)


# Defining Training Step
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for batch in iterator:
        src = batch[0]
        trg = batch[1]

        src = src.type(torch.LongTensor).to(device)
        trg = trg.type(torch.LongTensor).to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        print(loss.item())

        # Deleting Used Tensors
        # del src
        # del trg
        # gc.collect()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Defining Validation / Evaluation Step
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch[0]
            trg = batch[1]

            src = src.type(torch.LongTensor).to(device)
            trg = trg.type(torch.LongTensor).to(device)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            print(loss.item())

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Training Starts Here
N_EPOCHS = config.EPOCH
CLIP = 0.1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)

    print(f'Train Batch No.-{epoch + 1} Done')

    valid_loss = evaluate(model, valid_loader, criterion)

    print(f'Validation Batch No.-{epoch + 1} Done')

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss or epoch % 2 == 0:
        best_valid_loss = valid_loss
        if valid_loss < best_valid_loss and epoch % 2 != 0:
            torch.save(model.state_dict(), 'translate-model-eng-hin.pt')
        else:
            torch.save(model.state_dict(), 'translate-model-eng-hin-extra-train.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

