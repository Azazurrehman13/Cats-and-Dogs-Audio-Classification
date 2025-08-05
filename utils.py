import numpy as np
import pandas as pd
import scipy.io.wavfile as sci_wav
import random

ROOT_DIR = '/kaggle/input/audio-cats-and-dogs/cats_dogs/'
CSV_PATH = '/kaggle/input/audio-cats-and-dogs/train_test_split.csv'

def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths
    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]

def get_trunk(_X, idx, sample_len, rand_offset=False):
    '''Returns a trunk of the 1D array <_X>
    Params:
        _X: the concatenated audio samples
        idx: _X will be split in <sample_len> items. _X[idx]
        rand_offset: boolean to say whether or not we use an offset
    '''
    randint = np.random.randint(10000) if rand_offset else 0
    start_idx = (idx * sample_len + randint) % len(_X)
    end_idx = ((idx + 1) * sample_len + randint) % len(_X)
    if end_idx > start_idx:
        return _X[start_idx:end_idx]
    else:
        return np.concatenate((_X[start_idx:], _X[:end_idx]))

def get_augmented_trunk(_X, idx, sample_len, added_samples=0):
    X = get_trunk(_X, idx, sample_len)
    for _ in range(added_samples):
        ridx = np.random.randint(len(_X))
        X = X + get_trunk(_X, ridx, sample_len)
    return X

def dataset_gen(dataset, is_train=True, batch_shape=(20, 16000), sample_augmentation=0):
    '''This generator returns training batches of size <batch_shape>
    Params:
        dataset: Dictionary containing train_cat, train_dog, test_cat, test_dog
        is_train: True for training generator, False for test
        batch_shape: Tuple of (num_samples_per_batch, num_datapoints_per_sample)
        sample_augmentation: Number of additional audio samples to augment (train only)
    '''
    s_per_batch = batch_shape[0]
    s_len = batch_shape[1]
    X_cat = dataset['train_cat'] if is_train else dataset['test_cat']
    X_dog = dataset['train_dog'] if is_train else dataset['test_dog']
    nbatch = int(max(len(X_cat), len(X_dog)) / s_len)
    perms = [list(enumerate([i] * nbatch)) for i in range(2)]
    perms = sum(perms, [])
    random.shuffle(perms)
    y_batch = np.zeros(s_per_batch)
    X_batch = np.zeros(batch_shape)
    while len(perms) > s_per_batch:
        for bidx in range(s_per_batch):
            perm, _y = perms.pop()
            y_batch[bidx] = _y
            _X = X_cat if _y == 0 else X_dog
            if is_train:
                X_batch[bidx] = get_augmented_trunk(
                    _X, idx=perm, sample_len=s_len, added_samples=sample_augmentation)
            else:
                X_batch[bidx] = get_trunk(_X, perm, s_len)
        yield (X_batch.reshape(s_per_batch, s_len, 1),
               y_batch.reshape(-1, 1))

def load_dataset(dataframe):
    '''Load the dataset into a dictionary
    Params:
        dataframe: Pandas DataFrame with columns [train_cat, train_dog, test_cat, test_dog]
    Returns:
        dataset: Dictionary with keys train_cat, train_dog, test_cat, test_dog
    '''
    df = dataframe
    dataset = {}
    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        v = list(df[k].dropna())
        v = read_wav_files(v)
        v = np.concatenate(v).astype('float32')
        if k == 'train_cat':
            dog_std = dog_mean = 0
            cat_std, cat_mean = v.std(), v.mean()
        elif k == 'train_dog':
            dog_std, dog_mean = v.std(), v.mean()
        std, mean = (cat_std, cat_mean) if 'cat' in k else (dog_std, dog_mean)
        v = (v - mean) / std
        dataset[k] = v
        print(f'loaded {k} with {len(v) / 16000:.2f} sec of audio')
    return dataset
