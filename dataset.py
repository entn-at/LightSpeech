import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import Audio
from text import text_to_sequence
from utils import process_text, pad_1D, pad_2D
import hparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LightSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self):
        self.text = process_text(os.path.join("data", "train.txt"))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # mel_gt_name = os.path.join(
        #     hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (idx+1))
        # mel_gt_target = np.load(mel_gt_name)
        mel_tac2_target = np.load(os.path.join(
            hparams.mel_tacotron2, str(idx)+".npy")).T

        cemb = np.load(os.path.join(hparams.cemb_path, str(idx)+".npy"))
        D = np.load(os.path.join(hparams.alignment_path, str(idx)+".npy"))

        character = self.text[idx][0:len(self.text[idx])-1]
        character = np.array(text_to_sequence(
            character, hparams.text_cleaners))

        sample = {"text": character,
                  "mel_tac2_target": mel_tac2_target,
                  "cemb": cemb,
                  "D": D}

        return sample


def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    cembs = [batch[ind]["cemb"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]
    # mel_gt_targets = [batch[ind]["mel_gt_target"] for ind in cut_list]
    mel_tac2_targets = [batch[ind]["mel_tac2_target"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_mel = np.array(list())
    for mel in mel_tac2_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_1D(texts)
    Ds = pad_1D(Ds)
    # mel_gt_targets = pad_2D(mel_gt_targets)
    mel_tac2_targets = pad_2D(mel_tac2_targets)
    cembs = pad_2D(cembs)

    out = {"text": texts,
           "mel_tac2_target": mel_tac2_targets,
           "cemb": cembs,
           "D": Ds,
           "length_mel": length_mel,
           "length_text": length_text}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output
