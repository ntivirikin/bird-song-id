from pydub import AudioSegment
from pydub.utils import mediainfo
from scipy.io import wavfile
import errno
import shutil
import os
import random
import numpy
from datetime import datetime
from matplotlib import pyplot


# Print error messages
def err_log(error):
    date = str(datetime.utcnow())
    errmsg = '[' + date + '] ' + str(error) + ' occurred.'
    print(errmsg)


# Create directory safely
def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            err_log(e)
            raise


# Convert .mp3 to .wav files with desired sample rate
def resample_wav(src, dst, rate):
    create_dir(os.getcwd() + '/data/output')
    try:
        sound = AudioSegment.from_mp3(src)
    except FileNotFoundError as e:
        err_log(e)
        return
    sound = sound.set_frame_rate(rate)
    sound = sound.export(dst, format="wav")
    sound.close()


# Placeholder
def expand_ones(keep_cell):
    return 'Placeholder'


# Filters spectrogram based on energy signals
#
# Parameters:
# data              Audio data
# expand_one        Boolean, if True will apply the expand_by_one function
# drop_zero         Boolean, if True will apply zero cell elimination function
#
# Returns:
#
# unfilt_spec       Raw spectrogram array
# filt_spec         Filtered spectrogram array
def audio_spectro(data, expand_one = False, drop_zero = 0.95):

    # Calculating spectrogram with audio data
    raw_spec = numpy.log10(pyplot.specgram(data, NFFT=512, noverlap=256, Fs=16000)[0])

    # Cut off high frequencies
    # TODO: This needs to be tailored to each recording
    temp_spec = raw_spec[0:200,:]
    filter_spec = numpy.copy(temp_spec)

    # Split spectrogram into 20x30 arrays
    # TODO: Finetuning of array size
    row_bord = numpy.ceil(numpy.linspace(0, temp_spec.shape[0], 20))
    col_bord = numpy.hstack((numpy.ceil(numpy.arange(0, temp_spec.shape[1], 30)), temp_spec.shape[1]))

    # Casting to int
    row_bord = [ int(x) for x in row_bord ]
    col_bord = [ int(x) for x in col_bord ]
    keep_cell = numpy.ones((len(row_bord)-1, len(col_bord)-1))
    
    # Create 0 mask for spectrogram
    # Scans using 20x30 array creater earlier
    for i in range(len(row_bord)-1):

        # Mean and std for rows calculated for 0 mask creation
        row_mean = numpy.mean(temp_spec[row_bord[i]:row_bord[i+1],:])
        row_std = numpy.std(temp_spec[row_bord[i]:row_bord[i+1],:])
        
        # Mean for columns
        for j in range(len(col_bord) - 1):
            cell_mean = numpy.mean(temp_spec[row_bord[i]:row_bord[i+1],col_bord[j]:col_bord[j+1]])
            cell_max_top10_mean = numpy.mean(numpy.sort(temp_spec[row_bord[i]:row_bord[i+1],
                                             col_bord[j]:col_bord[j+1]], axis=None)[-10:])

            if (cell_mean < 0 or ((cell_max_top10_mean) < (row_mean + row_std)*1.5)):
                keep_cell[i, j] = 0

    # Expand by ones
    if expand_one:
        keep_cell = expand_ones(keep_cell)
    
    # Apply the mask to the spectrogram
    for i in range(keep_cell.shape[0]):
        for j in range(keep_cell.shape[1]):
            if not keep_cell[i, j]:
                filter_spec[row_bord[i]:row_bord[i+1],col_bord[j]:col_bord[j+1]] = 0

    # Drop zero columns
    # Amount of 0's in each column is calculated and dropped if higher than drop_zero
    filter_spec_backup = numpy.copy(filter_spec)
    filter_spec = numpy.delete(filter_spec, numpy.nonzero((filter_spec==0).sum(axis=0) > filter_spec.shape[0]*drop_zero), axis=1)

    # If all were 0 we can use the backup
    if filter_spec.shape[1] == 0:
        filter_spec = filter_spec_backup

    return temp_spec, filter_spec


# Processes entire path of .wav files and returns list of spectrograms
def process_wav(path, filenames, filt=False):
    data = []
    for name in filenames:
        (unfilt_spec, filt_spec) = audio_spectro(wavfile.read(os.path.join(path, name))[1], expand_one = True)
        if (not filt):
            data.append(filt_spec)
        else:
            data.append(unfilt_spec)
    return data


# Not sure if below is needed, keep for later
'''
# Sorts each species into a training, testing, and validating set
# TODO: Optimize this with set or bisect (needed?)
def create_struc(path):
    file_list = os.scandir(path)
    species_list = []
    fold_path = os.getcwd() + '/test_data/'

    for files in file_list:
       rec = files.name
       temp = rec.split('_')
       species = temp[0]
       if species not in species_list:
           species_list.append(species)


    for fold in species_list:
        make_path = fold_path + fold
        create_dir(make_path)
        create_dir(make_path + '/temp/')
        create_dir(make_path + '/temp/' + fold)
        create_dir(make_path + '/train/')
        create_dir(make_path + '/test/')
        create_dir(make_path + '/validate/')


def sort_rec(species_list, path, train = 0.6, test = 0.2, validate = 0.2):
    file_list = os.scandir(path)
    for files in file_list:
        rec = files.name
        temp = rec.split('_')
        struct = temp[0]
        
        for species in species_list:
            if struct == species:
                shutil.move(files.path,os.getcwd() +  '/temp/' + species)
    file_list.close()

    temp_list = os.scandir(os.getcwd() + '/temp' + species)

    for files in temp_list:
        num = random.randint(1,11)
        if num in range(1, (train * 10) + 1):
            # Move to train
            return None
        elif num in range(7, 9):
            # Move to test
            return None
        elif num in range (9, 11):
            # Move to validate
            return None
    temp_list.close()
'''
