import os
import warnings
import torch
from Model.train import model
from Model.train import evaluate, test_iterator, criterion, train_data, device
from Model.sentence_eval import translate_sentence
from Model.visulaize import display_attention
from Model.data_preprocess import SRC, TRG
import math

# For ignoring warnings
warnings.filterwarnings("ignore")


model.load_state_dict(torch.load('translate-model.pt'))

# test_loss = evaluate(model, test_iterator, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


# Testing
example_idx = 2

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')
display_attention(src, translation, attention)
