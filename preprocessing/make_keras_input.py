import numpy as np

from keras.utils import np_utils
from preprocessing.data_utils import get_ETL_data
from sklearn import datasets, metrics, cross_validation
from sklearn.utils import shuffle

def data(writers_per_char = 160, mode='all', get_scripts=False, test_size=0.2):
    """
    Load the characters into a format suitable for Keras
    Inputs:
    - writers_per_char: number of samples per Japanese character
    - mode: specify the type of Japanese characters: 'all', 'hiragana', 'kanji', or 'katakana'
    - get_scripts: 'True' returns a label for the type of script corresponding to each Japanese character
    - test_size: fraction of data to use for obtaining a test error
    
    Returns:
    - X_train: training data
    - Y_train: training labels
    - X_test: test data
    - Y_test: test labels
    """
    size = (64,64)
    if mode in ('kanji','all'):
        for i in range(1,4):
            if i == 3:
               max_records = 315
            else:
               max_records = 319

            if i != 1:
                start_record = 0
            else:
                start_record = 75

            if get_scripts:
                chars, labs, spts = get_ETL_data(i, range(start_record,max_records), writers_per_char, get_scripts = True)
            else:
                chars, labs = get_ETL_data(i, range(start_record,max_records), writers_per_char)
                
            if i == 1 and mode in ('kanji','all'):
                characters = chars
                labels = labs
                if get_scripts:
                    scripts = spts
            else:
                characters = np.concatenate((characters,chars), axis=0)
                labels = np.concatenate((labels,labs), axis=0)
                if get_scripts:
                    scripts = np.concatenate((scripts,spts), axis=0)

    if mode in ('hiragana','all'):
        max_records = 75
        if get_scripts:
            chars, labs, spts = get_ETL_data(1, range(0,max_records), writers_per_char, get_scripts = True)
        else:
            chars, labs = get_ETL_data(1, range(0,max_records), writers_per_char)

        if mode == 'hiragana':
            characters = chars
            labels = labs

        else:
            characters = np.concatenate((characters,chars), axis=0)
            labels = np.concatenate((labels,labs), axis=0)
            if get_scripts:
                scripts = np.concatenate((scripts,spts), axis=0)


    if mode in ('katakana','all'):
        for i in range(7,14):
            if i < 10:
                filename = '0'+str(i)
            else:
                filename = str(i)

            if get_scripts:
                chars, labs, spts = get_ETL_data(filename, range(0,8), writers_per_char, database='ETL1C', get_scripts=True)
            else:
                chars, labs = get_ETL_data(filename, range(0,8), writers_per_char, database='ETL1C')

            if i == 7 and mode == 'katakana':
                characters = chars
                labels = labs
            else:
                characters = np.concatenate((characters,chars), axis=0)
                labels = np.concatenate((labels,labs), axis=0)
                if get_scripts:
                    scripts = np.concatenate((scripts,spts), axis=0)

    # rename labels from 0 to n_labels-1
    unique_labels = list(set(labels))
    labels_dict = {unique_labels[i]:i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

    if get_scripts:
        characters_shuffle, scripts_shuffle = shuffle(characters, scripts, random_state=0)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(characters_shuffle,
                                                                             scripts_shuffle,
                                                                             test_size=test_size,
                                                                             random_state=42)
    elif mode in ('all','kanji','hiragana','katakana'):
        characters_shuffle, new_labels_shuffle = shuffle(characters, new_labels, random_state=0)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(characters_shuffle,
                                                                             new_labels_shuffle,
                                                                             test_size=test_size,
                                                                             random_state=42)


    # reshape to (1, 64, 64)
    X_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    X_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))

    # convert class vectors to binary class matrices
    nb_classes = len(unique_labels)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
