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
import pandas
import json
import h5py


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


# Loads metadata as a pandas dataframe
def load_meta(path='metadata.json'):
    open_t = open(path)
    df = pandas.read_json(open_t, orient='columns')
    return df


# Generate one-hot encoding for metadata
def ohot_encode(df, label):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(df[label])
    return (lb, lb.transform(df[label]))


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
# Return values:
# unfilt_spec       Raw spectrogram array
# filt_spec         Filtered spectrogram array
def audio_spectro(data, expand_one = False, drop_zero = 0.95):

    # Calculating spectrogram with audio data
    raw_spec = numpy.log10(pyplot.specgram(data, NFFT=512, noverlap=256, Fs=16000)[0])

    # Cut off high frequencies
    # TODO: This needs to be tailored to the dataset
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


# Creates training data from list of spectrograms
#
# Parameters:
# spec_list		Generated spectrogram list
# labels		Class label list
# N			    (1*44100)/(1024-512)=86
# filenames		Filename
# class_ids		ClassID
#
# Return values:
# X			    Input
# y			    Output
# filen			Filename
# c_id			ClassID
# TODO: Determine if class id's are required
def process_spec(spec_list, N, labels = None, filenames = None, class_ids = None):

    rows	= len(spec_list[0])
    X		= numpy.empty((0, 1, rows, N))
    y		= []
    filen 	= []
    c_id	= []

    for i in range(len(spec_list)):
        ranges = numpy.hstack((numpy.arange(0, len(spec_list[i][0]), N), len(spec_list[i][0])))

        for j in range(len(ranges) - 1):
            temp_spec = numpy.empty((1, rows, N))

            # Perform zero-padding if spectrogram is shorter than desired window length
            if (len(spec_list[i][0]) < N): 
            
                temp_spec[0] = numpy.hstack((spec_list[i], numpy.zeros((rows, N - len(spec_list[i][0])))))
            
            # Check if it is the last element
            elif (ranges[j + 1] - ranges[j] < N):
                temp_spec[0] = spec_list[i][:, -N:]
            else:
                temp_spec[0] = spec_list[i][:, ranges[j]:ranges[j + 1]]

            X = numpy.vstack((X, [temp_spec]))

            # Appending metadata to output lists
            if labels is not None:
                y.append(labels[i])
            if filenames is not None:
                filen.append(filenames[i])
            if class_ids is not None:
                c_id.append(class_ids[i])
    return X, y, filen, c_id


# Calculates the standard scalar coefficient of the input data for 0 mean and 1 variance
#
# Parameters:
# num_files 		the number of files to process (we assume that the mean and variance 
#                   will be similar in case of a subset of the training data and we don't
#                   have to process the whole database)
# wavdirpath		the path that contains the wavs (sampled at 16kHz)
# xmlpicklepath		the path and filename that contains the XML file for training (xml_data.pickle)
#
# Return values:
# scalar
# spectogramData
# TODO: Determine how to save scalars (Pickle, json, etc.)
def gen_scalar(wav_path, json_path, dirroot_path, num_files = 100):
    create_dir(dirroot_path)

    # Reading metadata with Pickle
    df = load_meta()

    # Shuffle rows
    df = df.iloc[numpy.random.permutation(len(df))]
    df.reset_index(drop = True, inplace = True)
    
    # Calculate spectograms
    spec_data = process_wav(wav_path, df.FileName[:num_files], filt = False)

    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()

    # Calculate scalar for each spectrogram
    for s_data in spec_data:
        scalar.partial_fit(s_data.reshape(-1, 1))

    # filename where we save the scaler
    save_to = os.path.join(dirroot_path, "standardScaler_{}.pickle".format(num_files))
    from sklearn.externals import joblib
    import pickle
    pickle.dump(scalar, open(save_to, 'wb'))
    print('Scaler saved to: {}'.format(save_to))
    
    return scalar, spec_data


# Constructs training data from recordings and metadata
#
# Parameters:
# wav_dir       Path to WAV recordings
# json_path     Path to JSON metadata file
# save_path     Path to save training data
#
# Returns:
#
# X, y, fn      For debugging if required
def gen_training(wav_dir, json_path=os.getcwd() + '/metadata.json', save_path=os.getcwd() + '/training/'):
    create_dir(save_path)
    # Setting some required variables
    # TODO: Grab these from the spectrogram processing, remove N entirely (process all classes in recordings)
    N = 100
    spectrogram_height = 100
    spectrogram_window_length = 100

    f = h5py.File(os.path.join(save_path,"data_top{}_nozero.hdf5".format(N)), "w")
    dsetX	= f.create_dataset('X', (0,1,spectrogram_height,spectrogram_window_length), maxshape=(None, 1,spectrogram_height,spectrogram_window_length))
    dsety	= f.create_dataset('y', (0,N), maxshape=(None,N))

    # Load json metadata
    df = load_meta()

    # Shuffle rows
    df = df.iloc[numpy.random.permutation(len(df))]
    df.reset_index(drop=True, inplace=True)

    # Generating one-hot labels
    (lb, binary_labels) = ohot_encode(df, "species")
    import pickle
    pickle.dump(lb, open(os.path.join(save_path,"labelBinarizer_top{}.pickle".format(N)), 'wb'))