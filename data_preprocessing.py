import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
from pydub import AudioSegment
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import shutil
from tempfile import TemporaryFile


ROOT_DIR = os.path.dirname(os.path.abspath(
    __file__))  # This is your Project Root


genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()


def make_dirs():
    os.makedirs(f'{ROOT_DIR}/cleaned/spectrograms5sec')
    os.makedirs(f'{ROOT_DIR}/cleaned/spectrograms5sec/train')
    os.makedirs(f'{ROOT_DIR}/cleaned/spectrograms5sec/test')
    for g in genres:
        path_audio = os.path.join(f'{ROOT_DIR}/cleaned/audio5sec', f'{g}')
        os.makedirs(path_audio)
        path = os.path.join(f'{ROOT_DIR}/cleaned/spectrograms5sec', f'{g}')
        path_train = os.path.join(
            f'{ROOT_DIR}/cleaned/spectrograms5sec/train', f'{g}')
        path_test = os.path.join(
            f'{ROOT_DIR}/cleaned/spectrograms5sec/test', f'{g}')
        os.makedirs(path)
        os.makedirs(path_train)
        os.makedirs(path_test)


def audio_split():
    for g in genres:
        j = 0
        print(f"{g}")
        for filename in os.listdir(os.path.join(f'{ROOT_DIR}/Data/genres_original', f"{g}")):
            song = os.path.join(
                f'{ROOT_DIR}/Data/genres_original/{g}', f'{filename}')
            for w in range(0, 6):
                t1 = 5*(w)*1000
                t2 = 5*(w+1)*1000
                # print(song)
                if (not song.endswith("jazz.00054.wav")):  # jazz.00054.wav is corrupt
                    newAudio = AudioSegment.from_wav(song)
                    new = newAudio[t1:t2]
                    new.export(
                        f'{ROOT_DIR}/cleaned/audio5sec/{g}/{g+str(j)}.wav', format="wav")
                    j = j + 1


# def make_spectrogram():
#     for g in genres:
#         j = 0
#         print(g)
#         for filename in os.listdir(os.path.join(f'{ROOT_DIR}/cleaned/audio5sec', f"{g}")):
#             song = os.path.join(
#                 f'{ROOT_DIR}/cleaned/audio5sec/{g}', f'{filename}')
#             j = j+1
#             y, sr = librosa.load(song, duration=3)
#             # print(sr)
#             mels = librosa.feature.melspectrogram(y=y, sr=sr)
#             np.save(f'{ROOT_DIR}/cleaned/spectrograms5sec/train/{g}/{g+str(j)}.npy', mels)
#             # fig = plt.Figure()
#             # canvas = FigureCanvas(fig)
#             # p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
#             # plt.savefig(
#             #     f'{ROOT_DIR}/cleaned/spectrograms5sec/train/{g}/{g+str(j)}.png')


# def split_data(train_percent=0.8):
#     dir = ROOT_DIR + "/cleaned/audio5sec/"
#     for g in genres:
#         filenames = os.listdir(os.path.join(dir, f'{g}'))
#         random.shuffle(filenames)
#         threshold = len(filenames) * (1-train_percent)
#         test_files = filenames[0:threshold]
#         for f in test_files:
#             shutil.move(
#                 dir + f"{g}" + "/" + f, f"{ROOT_DIR}/cleaned/spectrograms5sec/test/" + f"{g}")


if __name__ == "__main__":
    make_dirs()
    audio_split()
    make_spectrogram()  # saves to the train folder
    split_data()
