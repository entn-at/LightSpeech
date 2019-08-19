from torch import nn


class LigthSpeechLoss(nn.Module):
    """ LigthSpeech Loss """

    def __init__(self):
        super(LigthSpeechLoss, self).__init__()

    def forward(self, mel,
                padd_predicted, cemb_out,
                mel_tac2_target,
                D, cemb):
        mel_loss = nn.MSELoss()(mel, mel_tac2_target)

        # similarity_loss = nn.MSELoss()(cemb_out, cemb)
        similarity_loss = nn.L1Loss()(cemb_out, cemb)

        duration_loss = nn.L1Loss()(padd_predicted, D.float())

        return mel_loss, similarity_loss, duration_loss
