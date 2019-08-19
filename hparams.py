from text import symbols

# Text
text_cleaners = ['english_cleaners']

# Mel
n_mel_channels = 80

# LightSpeech
embedding_size = len(symbols)
embedding_dim = 256


pre_gru_in_dim = 256
pre_gru_out_dim = 512
pre_gru_layer_size = 1

post_gru_in_dim = 512
post_gru_out_dim = 256
post_gru_layer_size = 1

# Duration Predictor
dp_filter_size = 256
dp_kernel_size = 3
dp_dropout = 0.1

# Train
cemb_path = "./cembs"
alignment_path = "./alignments"
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_tacotron2 = "./mel_tac2"

batch_size = 64
epochs = 1000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 1000
log_step = 5
clear_Time = 20
