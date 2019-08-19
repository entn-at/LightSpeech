import torch
import torch.nn as nn
from torch import optim
from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time

from networks import LightSpeech
from dataset import DataLoader, collate_fn
from dataset import LightSpeechDataset
from loss import LigthSpeechLoss
import utils
import hparams as hp


def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = nn.DataParallel(LightSpeech()).to(device)
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of LightSpeech Parameters:', num_param)

    # Get dataset
    dataset = LightSpeechDataset()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp.learning_rate,
                                 weight_decay=hp.weight_decay)

    # Criterion
    criterion = LigthSpeechLoss()

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)

    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Define Some Information
    Time = np.array([])
    Start = time.clock()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        # Get Training Loader
        training_loader = DataLoader(dataset,
                                     batch_size=hp.batch_size**2,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True,
                                     num_workers=0)
        total_step = hp.epochs * len(training_loader) * hp.batch_size

        for i, batchs in enumerate(training_loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.clock()

                current_step = i * hp.batch_size + j + args.restore_step + \
                    epoch * len(training_loader)*hp.batch_size + 1

                # Init
                optimizer.zero_grad()

                # Get Data
                character = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                # mel_gt_target = torch.from_numpy(
                #     data_of_batch["mel_gt_target"]).float().to(device)
                mel_tac2_target = torch.from_numpy(
                    data_of_batch["mel_tac2_target"]).float().to(device)

                D = torch.from_numpy(data_of_batch["D"]).int().to(device)
                cemb = torch.from_numpy(
                    data_of_batch["cemb"]).float().to(device)

                input_lengths = torch.from_numpy(
                    data_of_batch["length_text"]).long().to(device)
                output_lengths = torch.from_numpy(
                    data_of_batch["length_mel"]).long().to(device)

                max_c_len = max(input_lengths).item()
                max_mel_len = max(output_lengths).item()

                # Forward
                mel, padd_predicted, cemb_out = model(character,
                                                      input_lengths,
                                                      output_lengths,
                                                      max_c_len,
                                                      max_mel_len, D)

                # Cal Loss
                mel_loss, similarity_loss, duration_loss = criterion(mel,
                                                                     padd_predicted, cemb_out,
                                                                     mel_tac2_target,
                                                                     D, cemb)
                total_loss = mel_loss + similarity_loss + duration_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                s_l = similarity_loss.item()
                d_l = duration_loss.item()

                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")

                with open(os.path.join("logger", "similarity_loss.txt"), "a") as f_s_loss:
                    f_s_loss.write(str(s_l)+"\n")

                with open(os.path.join("logger", "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), 1.)

                # Update weights
                optimizer.step()

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.clock()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Mel Loss: {:.4f}, Duration Loss: {:.4f}, Similarity Loss: {:.4f};".format(
                        m_l, d_l, s_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.clock()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
