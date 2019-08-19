import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import glow
import waveglow

from networks import LightSpeech
from text import text_to_sequence
import hparams as hp
import utils
import Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_LightSpeech(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(LightSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path))['model'])
    model.eval()

    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    text = np.stack([text])
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).cuda().long()
        # mel, mel_postnet_1, mel_postnet_2 = model.module.inference(
        #     sequence, alpha)
        mel = model.module.inference(sequence, alpha)

        # out = mel[0].cpu().transpose(0, 1),\
        #     mel_postnet_1[0].cpu().transpose(0, 1),\
        #     mel_postnet_2[0].cpu().transpose(0, 1),\
        #     mel.transpose(1, 2),\
        #     mel_postnet_1.transpose(1, 2),\
        #     mel_postnet_2.transpose(1, 2)

        return mel[0].cpu().transpose(0, 1), mel.transpose(1, 2)


if __name__ == "__main__":
    # Test
    num = 178000
    model = get_LightSpeech(num)
    words = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"

    # mel, mel_postnet_1, mel_postnet_2, \
    #     mel_torch, mel_postnet_1_torch, mel_postnet_2_torch = synthesis(
    #         model, words, alpha=1.0)

    mel, mel_torch = synthesis(model, words, alpha=1.0)

    if not os.path.exists("results"):
        os.mkdir("results")
    Audio.tools.inv_mel_spec(mel, os.path.join(
        "results", words + str(num) + "griffin_lim.wav"))

    wave_glow = utils.get_WaveGlow()
    waveglow.inference.inference(mel_torch, wave_glow, os.path.join(
        "results", words + str(num) + "waveglow.wav"))
    # waveglow.inference.inference(mel_postnet_1_torch, wave_glow, os.path.join(
    #     "results", words + str(num) + "postnet_1_waveglow.wav"))
    # waveglow.inference.inference(mel_postnet_2_torch, wave_glow, os.path.join(
    #     "results", words + str(num) + "postnet_2_waveglow.wav"))

    # utils.plot_data(
    #     [mel.numpy(), mel_postnet_1.numpy(), mel_postnet_2.numpy()])
    # utils.plot_data([mel.numpy() for _ in range(2)])

    tacotron2 = utils.get_Tacotron2()
    mel_tac2, _, _ = utils.load_data_from_tacotron2(words, tacotron2)
    waveglow.inference.inference(torch.stack([torch.from_numpy(
        mel_tac2).cuda()]), wave_glow, os.path.join("results", "tacotron2.wav"))

    utils.plot_data([mel.numpy(), mel_tac2])

    texts = utils.process_text(os.path.join("data", "train.txt"))
    txts = list()
    for txt in texts:
        text = np.array(text_to_sequence(txt, hp.text_cleaners))
        text = np.stack([text])
        with torch.no_grad():
            sequence = torch.autograd.Variable(
                torch.from_numpy(text)).cuda().long()
            txts.append(sequence)

    cnt = 0
    for t in txts[0:100]:
        cnt += 1
        if cnt % 10 == 0:
            print("Done", cnt)
        model.module.inference(t)

    test = txts[5000:5100]
    length = len(test)
    start_time = time.clock()
    for i in range(length):
        mel = model.module.inference(test[i])
        waveglow.inference.test_speed(mel.transpose(1, 2), wave_glow)
    end_time = time.clock()
    print((end_time-start_time) / length)
