import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import Linear, LengthRegulator, PostNet
import hparams
import utils

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class LightSpeech(nn.Module):
    """ LightSpeech """

    def __init__(self):
        super(LightSpeech, self).__init__()
        self.embeddings = nn.Embedding(hparams.embedding_size+1,
                                       hparams.embedding_dim,
                                       padding_idx=0)

        self.pre_gru = nn.GRU(hparams.pre_gru_in_dim,
                              int(hparams.pre_gru_out_dim / 2),
                              num_layers=hparams.pre_gru_layer_size,
                              batch_first=True,
                              bidirectional=True)
        self.pre_linear = Linear(hparams.pre_gru_out_dim,
                                 hparams.post_gru_in_dim)
        self.LR = LengthRegulator()

        self.post_gru = nn.GRU(hparams.post_gru_in_dim,
                               int(hparams.post_gru_out_dim / 2),
                               num_layers=hparams.post_gru_layer_size,
                               batch_first=True,
                               bidirectional=True)
        self.post_linear = Linear(hparams.post_gru_out_dim,
                                  hparams.n_mel_channels)

        # self.postnet = PostNet()

    def pad_again(self, x, max_len):
        return F.pad(x, (0, 0, 0, max_len-x.size(1)))

    def mask(self, x, predicted, cemb, input_lengths, output_lengths, max_c_len, max_mel_len):
        x_mask = ~utils.get_mask_from_lengths(output_lengths, max_mel_len)
        x_mask = x_mask.expand(hparams.n_mel_channels,
                               x_mask.size(0), x_mask.size(1))
        x_mask = x_mask.permute(1, 2, 0)

        c_mask = ~utils.get_mask_from_lengths(input_lengths, max_c_len)
        c_mask = c_mask.expand(hparams.pre_gru_out_dim,
                               c_mask.size(0), c_mask.size(1))
        c_mask = c_mask.permute(1, 2, 0)

        x.data.masked_fill_(x_mask, 0.0)
        cemb.data.masked_fill_(c_mask, 0.0)
        predicted.data.masked_fill_(c_mask[:, :, 0], 0.0)

        return x, predicted, cemb

    def forward(self, character, length_c, length_mel, max_c_len, max_mel_len, D=None):
        x = self.embeddings(character)

        length_c_copy = length_c.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, length_c_copy, batch_first=True)
        self.pre_gru.flatten_parameters()
        x, _ = self.pre_gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.pad_again(x, max_c_len)

        cemb = self.pre_linear(x)

        x, predicted = self.LR(cemb, target=D, mel_max_length=max_mel_len)

        sorted_length_index = np.argsort(-length_mel.cpu().numpy())
        sorted_length_mel = np.array(list())
        x_sorted = list()

        for ind in sorted_length_index:
            x_sorted.append(x[ind])
            sorted_length_mel = np.append(
                sorted_length_mel, length_mel[ind].item())

        x_sorted = torch.stack(x_sorted).to(device)
        x_sorted = nn.utils.rnn.pack_padded_sequence(x_sorted,
                                                     sorted_length_mel,
                                                     batch_first=True)
        self.post_gru.flatten_parameters()
        x_sorted, _ = self.post_gru(x_sorted)
        x_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            x_sorted, batch_first=True)
        x_sorted = self.pad_again(x_sorted, max_mel_len)
        x = list([0 for _ in range(x_sorted.size(0))])

        for i, ind in enumerate(sorted_length_index):
            x[ind] = x_sorted[i]
        x = torch.stack(x).to(device)

        x = self.post_linear(x)
        mel, predicted, cemb = self.mask(
            x, predicted, cemb, length_c, length_mel, max_c_len, max_mel_len)

        # mel_postnet_1, mel_postnet_2 = self.postnet(
        #     mel, length_mel, max_mel_len)

        # return [mel, mel_postnet_1, mel_postnet_2], predicted, cemb
        return mel, predicted, cemb

    def inference(self, character, alpha=1.0):
        x = self.embeddings(character)

        self.pre_gru.flatten_parameters()
        x, _ = self.pre_gru(x)

        x = self.pre_linear(x)
        x = self.LR(x, alpha=alpha)

        self.post_gru.flatten_parameters()
        x, _ = self.post_gru(x)
        mel = self.post_linear(x)
        # mel_postnet_1, mel_postnet_2 = self.postnet.inference(mel)

        # return mel, mel_postnet_1, mel_postnet_2
        return mel


if __name__ == "__main__":
    # Test
    num_1 = utils.get_param_num(LightSpeech())
    print(num_1)

    model = utils.get_Tacotron2()
    num_2 = utils.get_param_num(model)
    print(num_2 / num_1)
