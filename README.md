# LightSpeech
A Light, Fast and Robust Speech Synthesis.

## Architecture
I use an embedding layer whose dimension size is 256 and a one layer GRU whose input dimension is 256 and output dimension is 512 as encoder. I also use a length regulator module to expand encoder output sequence to make its length equal to mel spectrogram's which shares the same idea as [FastSpeech](https://arxiv.org/abs/1905.09263), however, I  use two layer linear network to replace the conv layer proposed in the FastSpeech. I observed that two layer linear net could learn the alignment between encoder output and mel spectrogram which has less parameter than conv layer. In the decoder, I use a one layer GRU whose input dimension is 512 and output dimension is 256. To synthesize mel spectrogram, I use a linear layer as post net to output 80 dimension mel spectrogram. **The number of total parameter is just 1.8M which is 1/19 of FastSpeech's and 1/15 of Tacotron2's.**

## Training
I use Tacotron2 which is in the inference stage to provide alignment and mel spectrogram as targets of training LigthSpeech. I find that if I use Tacotron2 which is in the training stage to provide alignment as target of length regulator and ground truth mel spectrogram as target of decoder, it will cause the model performs worse. **I don't know why, and above-mentioned will be a bottleneck of the performance of LightSpeech.**

## How to Use
1. Put [Nvidia pretrained Tacotron2 model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing) in the `Tacotron2/pre_trained_model`;
2. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in the `waveglow/pre_trained_model`;
3. `python preprocess`;
4. `python train.py`.

## Eval
Here is [result](https://github.com/xcmyz/LightSpeech/tree/master/results). **LightSpeech is faster than FastSpeech which is 0.020s/6.7s and much lighter than FastSpeech.**

## Weakness
The quality of audio synthesized.
