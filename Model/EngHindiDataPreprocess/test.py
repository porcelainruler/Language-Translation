import os
import warnings
import torch
from Model.EngHindiDataPreprocess.train import model, device
from Model.EngHindiDataPreprocess.eng_hin_vocab_creator import ENG_DATA, HIN_DATA, hin_vocab_intToText,\
    eng_vocab_intToText
from Model.EngHindiDataPreprocess.sentence_eval import translate_sentence
from Model.EngHindiDataPreprocess import config
from Model.EngHindiDataPreprocess.visulaize import display_attention
import math

# For ignoring warnings
warnings.filterwarnings("ignore")


model.load_state_dict(torch.load('translate-model-eng-hin-extra-train.pt'))

# Checking Test Loss   <Test Set Not Defined so currently Not Working>
# test_loss = evaluate(model, test_iterator, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Testing
example_idx = 2

src = ENG_DATA[5]
trg = HIN_DATA[5]

# custom_src = ['']

print(f'src = {src}')
print(f'trg = {trg}')

# exit(0)


def convert_pred_trg_to_token(pred_trg: list, flag: bool):
    tokens = list()

    for tok in pred_trg:
        if flag:
            if tok in hin_vocab_intToText.keys():
                tokens.append(hin_vocab_intToText[tok])
            else:
                tokens.append(config.UNK_TOKEN.lower())
        else:
            if tok in eng_vocab_intToText.keys():
                tokens.append(eng_vocab_intToText[tok])
            else:
                tokens.append(config.UNK_TOKEN.lower())

    return tokens


translation, attention = translate_sentence(src, model, device)

src_text_form = convert_pred_trg_to_token(src[1:], False)
print(f'original trg = {src_text_form}')

trg_text_form = convert_pred_trg_to_token(trg[1:], True)
print(f'original trg = {trg_text_form}')

translation = convert_pred_trg_to_token(translation, True)
print(f'predicted trg = {translation}')

# Attention Visualization:
display_attention(src_text_form, translation, attention)
