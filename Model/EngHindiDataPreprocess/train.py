import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

from torchtext.data import BucketIterator

import random
import math
import time

# Import Data Pre-process Files and Config File
# from Model.CNNSeq2Seq import data_preprocess as dp
from Model.CNNSeq2Seq import config
from Model.CNNSeq2Seq import model
from Model.EngHindiDataPreprocess.eng_hin_vocab_creator import ENG_DATA, HIN_DATA

from sklearn.model_selection import train_test_split

import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

# Setting Parameters
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

BATCH_SIZE = config.BATCH_SIZE

np_eng = np.array(ENG_DATA)
np_hin = np.array(HIN_DATA)

tensor_eng = torch.tensor(ENG_DATA, dtype=torch.int32)
tensor_hin = torch.tensor(HIN_DATA, dtype=torch.int32)

X_train, X_val, y_train, y_val = train_test_split(tensor_eng, tensor_hin, test_size=0.1)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, drop_last=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, drop_last=True, num_workers=0)

print(len(X_train), len(X_val))
print(len(y_train), len(y_val))

# print(X_val)
print('-----------------------------------------------------------------------------')
# print(train_dataset[0])

count = 0
for batch in valid_loader:
    count += 1
    print(batch)

print(count, len(X_val)//config.BATCH_SIZE, len(X_train)//config.BATCH_SIZE)
exit(0)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      device=device)

SRC, TRG = dp.get_vocab()

print()
for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg
    print(src, trg)

exit(0)

# Setting up Model Paramters and Initializing the Model
INPUT_DIM = config.INPUT_DIM
OUTPUT_DIM = config.OUTPUT_DIM
EMB_DIM = config.EMB_DIM
HID_DIM = config.HID_DIM  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = config.ENC_LAYERS  # number of conv. blocks in encoder
DEC_LAYERS = config.DEC_LAYERS  # number of conv. blocks in decoder
ENC_KERNEL_SIZE = config.ENC_KERNEL_SIZE  # must be odd!
DEC_KERNEL_SIZE = config.DEC_KERNEL_SIZE  # can be even or odd
ENC_DROPOUT = config.ENC_DROPOUT
DEC_DROPOUT = config.DEC_DROPOUT
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

enc = model.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = model.Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = model.CNNSeq2Seq(enc, dec).to(device)


# For checking num. of trainable parameters in Model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

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

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

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


'''
# Training Starts Here
N_EPOCHS = config.EPOCH
CLIP = 0.1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'translate-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
'''
