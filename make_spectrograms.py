import torchaudio
from torchaudio import transforms
import os
import numpy as np

genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()

ROOT_DIR = os.path.dirname(os.path.abspath(
    __file__))  # This is your Project Root


for genre in genres:
    i = 0
    for filename in os.listdir(os.path.join(f'{ROOT_DIR}/cleaned/audio5sec', f"{genre}")):
        filename = f'{ROOT_DIR}/cleaned/audio5sec/{genre}/{filename}'
        waveform, sample_rate = torchaudio.load(filename, normalize=True)
        transform = transforms.MelSpectrogram(sample_rate)
        mel_specgram = transform(waveform)  # (channel, n_mels, time)
        save_file = f'{ROOT_DIR}/cleaned/spectrograms5sec/{genre}/{genre}{i}'
        np.save(save_file, mel_specgram)
        i = i + 1

