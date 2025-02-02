import torch
from torch import nn
import sentencepiece as sp
import torch.nn.functional as F
import random
import numpy as np

source_tokenizer = sp.SentencePieceProcessor()
target_tokenizer = sp.SentencePieceProcessor()

source_tokenizer.load("C://Users/Phone Myint Naing/Documents/AIT/classes/Natural Language Processing/Assignments/NLP-A3-st124973/translation_models/burmese_tokenizer.model")
target_tokenizer.load("C://Users/Phone Myint Naing/Documents/AIT/classes/Natural Language Processing/Assignments/NLP-A3-st124973/translation_models/english_tokenizer.model")


class Seq2SeqPackedAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        # src: [src len, batch_size]
        mask = (src == self.src_pad_idx).permute(1, 0)  # permute so that it's the same shape as attention
        # mask: [batch_size, src len] #(0, 0, 0, 0, 0, 1, 1)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src: [src len, batch_size]
        # trg: [trg len, batch_size]

        # initialize something
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)

        # send our src text into encoder
        encoder_outputs, hidden = self.encoder(src, src_len)
        # encoder_outputs refer to all hidden states (last layer)
        # hidden refer to the last hidden state (of each layer, of each direction)

        input_ = trg[0, :]

        mask = self.create_mask(src)  # (0, 0, 0, 0, 0, 1, 1)

        # for each of the input of the trg text
        for t in range(1, trg_len):
            # send them to the decoder
            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)
            # output: [batch_size, output_dim] ==> predictions
            # hidden: [batch_size, hid_dim]
            # attention: [batch_size, src len]

            # append the output to a list
            outputs[t] = output
            attentions[t] = attention

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # autoregressive

            input_ = trg[t] if teacher_force else top1

        return outputs, attentions


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # embedding
        embedded = self.dropout(self.embedding(src))
        # packed
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        # rnn
        packed_outputs, hidden = self.rnn(packed_embedded)
        # unpacked
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # -1, -2 hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs: [src len, batch_size, hid dim * 2]
        # hidden:  [batch_size, hid_dim]

        return outputs, hidden


class GeneralAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        # Add a linear layer to project encoder_outputs to the same dimensionality as hidden
        self.projection = nn.Linear(hid_dim * 2, hid_dim, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, hid_dim] (decoder hidden state)
        # encoder_outputs: [src_len, batch_size, hid_dim * 2] (encoder outputs)
        # mask: [batch_size, src_len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Project encoder_outputs to [src_len, batch_size, hid_dim]
        projected_encoder_outputs = self.projection(encoder_outputs)  # [src_len, batch_size, hid_dim]

        # Reshape hidden and projected_encoder_outputs for dot product
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hid_dim]
        projected_encoder_outputs = projected_encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim]

        # Compute alignment scores (dot product)
        energy = torch.matmul(hidden, projected_encoder_outputs.permute(0, 2, 1))  # [batch_size, 1, src_len]
        energy = energy.squeeze(1)  # [batch_size, src_len]

        # Apply mask
        energy = energy.masked_fill(mask, -1e10)

        # Compute attention weights
        attention = F.softmax(energy, dim=1)  # [batch_size, src_len]

        return attention


class MultiplicativeAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.W = nn.Linear(hid_dim * 2, hid_dim, bias=False)  # Weight matrix W

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, hid_dim] (decoder hidden state)
        # encoder_outputs: [src_len, batch_size, hid_dim * 2] (encoder outputs)
        # mask: [batch_size, src_len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Reshape hidden and encoder_outputs
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]

        # Apply weight matrix W to encoder_outputs
        transformed = self.W(encoder_outputs)  # [batch_size, src_len, hid_dim]

        # Compute alignment scores (multiplicative attention)
        energy = torch.matmul(hidden, transformed.permute(0, 2, 1))  # [batch_size, 1, src_len]
        energy = energy.squeeze(1)  # [batch_size, src_len]

        # Apply mask
        energy = energy.masked_fill(mask, -1e10)

        # Compute attention weights
        attention = F.softmax(energy, dim=1)  # [batch_size, src_len]

        return attention


class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.W1 = nn.Linear(hid_dim * 2, hid_dim, bias=False)  # Weight matrix W1 for encoder_outputs
        self.W2 = nn.Linear(hid_dim, hid_dim, bias=False)  # Weight matrix W2 for decoder hidden state
        self.v = nn.Linear(hid_dim, 1, bias=False)  # Weight vector v

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, hid_dim] (decoder hidden state)
        # encoder_outputs: [src_len, batch_size, hid_dim * 2] (encoder outputs)
        # mask: [batch_size, src_len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Reshape hidden and encoder_outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]

        # Compute additive attention scores
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))  # [batch_size, src_len, hid_dim]
        energy = self.v(energy).squeeze(2)  # [batch_size, src_len]

        # Apply mask
        energy = energy.masked_fill(mask, -1e10)

        # Compute attention weights
        attention = F.softmax(energy, dim=1)  # [batch_size, src_len]

        return attention


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch_size]
        # hidden: [batch_size, hid_dim]
        # encoder_ouputs: [src len, batch_size, hid_dim * 2]
        # mask: [batch_size, src len]

        # embed our input
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]

        # calculate the attention
        a = self.attention(hidden, encoder_outputs, mask)
        # a = [batch_size, src len]
        a = a.unsqueeze(1)
        # a = [batch_size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_ouputs: [batch_size, src len, hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted: [batch_size, 1, hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, batch_size, hid_dim * 2]

        # send the input to decoder rnn
        # concatenate (embed, weighted encoder_outputs)
        # [1, batch_size, emb_dim]; [1, batch_size, hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, batch_size, emb_dim + hid_dim * 2]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # send the output of the decoder rnn to fc layer to predict the word
        # prediction = fc(concatenate (output, weighted, embed))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc(torch.cat((embedded, output, weighted), dim=1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


input_dim = len(source_tokenizer)
output_dim = len(target_tokenizer)
emb_dim = 256
hid_dim = 512
dropout = 0.5
lr = 0.001
SRC_PAD_IDX = source_tokenizer.pad_id()
device = 'cpu'


# general_attention
gen_attn = GeneralAttention(hid_dim)
enc = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
gen_dec = Decoder(output_dim, emb_dim,  hid_dim, dropout, gen_attn)
gen_model = Seq2SeqPackedAttention(enc, gen_dec, SRC_PAD_IDX, device).to(device)
gen_model.load_state_dict(torch.load('C://Users/Phone Myint Naing/Documents/AIT/classes/Natural Language Processing/Assignments/NLP-A3-st124973/translation_models/general_attention.pt', map_location=torch.device('cpu')))


# multiplicative_attention
mul_attn = MultiplicativeAttention(hid_dim)
enc = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
mul_dec = Decoder(output_dim, emb_dim,  hid_dim, dropout, mul_attn)
mul_model = Seq2SeqPackedAttention(enc, mul_dec, SRC_PAD_IDX, device).to(device)
mul_model.load_state_dict(torch.load('C://Users/Phone Myint Naing/Documents/AIT/classes/Natural Language Processing/Assignments/NLP-A3-st124973/translation_models/multiplicative_attention.pt', map_location=torch.device('cpu')))

# additive_attention
add_attn = AdditiveAttention(hid_dim)
enc = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
add_dec = Decoder(output_dim, emb_dim,  hid_dim, dropout, add_attn)
add_model = Seq2SeqPackedAttention(enc, add_dec, SRC_PAD_IDX, device).to(device)
add_model.load_state_dict(torch.load('C://Users/Phone Myint Naing/Documents/AIT/classes/Natural Language Processing/Assignments/NLP-A3-st124973/translation_models/additive_attention.pt', map_location=torch.device('cpu')))


def translate_sentence(sentence, model, src_tokenizer, trg_tokenizer, device=device, max_len=50):
    model.eval()

    # Tokenize input sentence using SentencePiece
    tokens = [src_tokenizer.bos_id()] + src_tokenizer.encode(sentence, out_type=int) + [src_tokenizer.eos_id()]

    # Convert tokens to tensor
    src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)  # Shape: [src_len, 1]
    src_len = torch.tensor([len(tokens)]).to(device)

    # Initialize target sequence with <sos>
    trg_indexes = [trg_tokenizer.bos_id()]
    attentions = []  # Store attention weights

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)  # Last predicted token

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs,
                                                      model.create_mask(src_tensor))

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        attentions.append(attention.squeeze(1).cpu().numpy())  # Convert to NumPy for visualization

        if pred_token == trg_tokenizer.eos_id():  # Stop at <eos>
            break

    # Convert token IDs back to words
    translation = trg_tokenizer.decode(trg_indexes[1:-1])  # Skip <sos> and <eos>

    return translation, np.array(attentions)