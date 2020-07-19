import torch
import spacy
from Model.EngHindiDataPreprocess import config, eng_hin_vocab_creator as vocab
import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")


def translate_sentence(sentence, model, device, max_len=50):
    model.eval()

    tokens = [token for token in sentence]

    # src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [config.SOS_TOKEN_IDX]
    # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == config.EOS_TOKEN_IDX:
            break

    trg_tokens = trg_indexes

    return trg_tokens[1:], attention