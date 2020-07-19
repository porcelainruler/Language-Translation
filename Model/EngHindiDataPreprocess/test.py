import os
import warnings
import torch
from Model.EngHindiDataPreprocess.train import model, device
from Model.EngHindiDataPreprocess.eng_hin_vocab_creator import ENG_DATA, HIN_DATA
from Model.EngHindiDataPreprocess.sentence_eval import translate_sentence
from Model.CNNSeq2Seq.visulaize import display_attention
import math

# For ignoring warnings
warnings.filterwarnings("ignore")


model.load_state_dict(torch.load('translate-model-eng-hin-extra-train.pt'))

# Checking Test Loss   <Test Set Not Defined so currently Not Working>
# test_loss = evaluate(model, test_iterator, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Testing
example_idx = 2

src = ENG_DATA[0]
trg = HIN_DATA[0]

print(f'src = {src}')
print(f'trg = {trg}')

# exit(0)

translation, attention = translate_sentence(src, model, device)

print(f'predicted trg = {translation}')

# To be Made:
# display_attention(src, translation, attention)
