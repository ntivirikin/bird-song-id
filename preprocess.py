from pydub import AudioSegment
from pydub.utils import mediainfo
import errno
import os

# Create directory safely
def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Convert .mp3 to .wav files with desired sample rate
def resample_wav(src, dst, rate):
    create_dir(os.getcwd() + '/data/output')
    sound = AudioSegment.from_mp3(src)
    sound = sound.set_frame_rate(rate)
    sound = sound.export(dst, format="wav")
    sound.close()
    return mediainfo(dst)['sample_rate']