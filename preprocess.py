import torch
import numpy as np
import os

from utils import load_data, get_Tacotron2, get_WaveGlow
from utils import process_text, load_data_from_tacotron2
from data import ljspeech
import hparams as hp
import waveglow
import Audio


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = hp.mel_ground_truth
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


def main():
    # path = os.path.join("data", "LJSpeech-1.1")
    # preprocess_ljspeech(path)

    text_path = os.path.join("data", "train.txt")
    texts = process_text(text_path)

    if not os.path.exists(hp.cemb_path):
        os.mkdir(hp.cemb_path)

    if not os.path.exists(hp.alignment_path):
        os.mkdir(hp.alignment_path)

    if not os.path.exists(hp.mel_tacotron2):
        os.mkdir(hp.mel_tacotron2)

    tacotron2 = get_Tacotron2()
    # wave_glow = get_WaveGlow()

    num = 0
    for ind, text in enumerate(texts[num:]):
        print(ind)
        # mel_name = os.path.join(hp.mel_ground_truth,
        #                         "ljspeech-mel-%05d.npy" % (ind+1))
        # mel_target = np.load(mel_name)
        character = text[0:len(text)-1]
        mel_tacotron2, cemb, D = load_data_from_tacotron2(character, tacotron2)

        np.save(os.path.join(hp.mel_tacotron2, str(ind+num) + ".npy"),
                mel_tacotron2, allow_pickle=False)
        np.save(os.path.join(hp.cemb_path, str(ind+num) + ".npy"),
                cemb, allow_pickle=False)
        np.save(os.path.join(hp.alignment_path, str(
            ind+num) + ".npy"), D, allow_pickle=False)


if __name__ == "__main__":
    main()
