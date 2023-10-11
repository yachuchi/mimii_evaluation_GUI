#!/usr/bin/env python
"""
 @file   evaluate.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""

########################################################################
# How to run evaluate.py
########################################################################
"""
Environment : mobaxtern 
command : python3 evaluate.py
"""

########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
import torch
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
import tkinter as tk
# from import
from tqdm import tqdm
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import accuracy_score
from tkinter import filedialog
from PIL import  ImageTk, Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################

########################################################################
'''This part is for cmd test'''
# parser = argparse.ArgumentParser()
# parser.add_argument('--file_path', default="./dataset/0dB/fan/id_00/abnormal/00000001.wav", type=str)
# args = parser.parse_args()
# filepath = args.file_path

########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################

########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    
    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################

####################################################################
#                      button action                               #
####################################################################

def loadFile():
    global filepath
    filepath = filedialog.askopenfilename(filetypes = [("all files","*.*")])
    loadFile_en.delete(0 ,'end')
    loadFile_en.insert(0,filepath)

def loadModel():
    global modelpath
    modelpath = filedialog.askopenfilename(filetypes = [("all files","*.*")])
    loadModel_en.delete(0 ,'end')
    loadModel_en.insert(0,modelpath)

def eval():
    #clear all entry
    testresult_en.delete(0 ,'end')

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)
    
    # load filepath list
    db = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(filepath)[0])[0])[0])[0])[1]
    machine_type = os.path.split(os.path.split(os.path.split(os.path.split(filepath)[0])[0])[0])[1]
    machine_id = os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1]
    aorn = os.path.split(os.path.split(filepath)[0])[1]
    filename = os.path.split(filepath)[1]

    # load modelpath list
    model_type = os.path.split(modelpath)[1]
    # print(model_type[0:12])
    model_file = modelpath

    train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id, db=db)
    
    print("============== LOAD MODEL ==============")
    train_data = load_pickle(train_pickle)
    model = torch.load(model_file)

    print("============== EVALUATION ==============")
    y_pred = []
    if model_type[0:12] == "unsupervised":
        data = file_to_vector_array(filepath,
                                n_mels=param["feature"]["n_mels"],
                                frames=param["feature"]["frames"],
                                n_fft=param["feature"]["n_fft"],
                                hop_length=param["feature"]["hop_length"],
                                power=param["feature"]["power"])
        error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
        y_pred= numpy.mean(error)
        train_error = numpy.mean(numpy.square(train_data - model.predict(train_data)), axis=1)
        train_avg_error = numpy.mean(train_error)
        # print(f'train_avg_error:{train_avg_error}')
        # print(f'y_pred:{y_pred}')
        if y_pred > train_avg_error:
            testresult_en.insert(0,"abnormal")
        else:
            testresult_en.insert(0,"normal")
    elif model_type[0:10] == "supervised":
        data = file_to_vector_array(filepath,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"])
        error = model.predict(data)
        y_pred= numpy.round(numpy.mean(model.predict(data)))
        # print(f'y_pred:{y_pred}')
        if y_pred == 1.0:
            testresult_en.insert(0,"abnormal")
        elif y_pred == 0:
            testresult_en.insert(0,"normal")
    
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.subplot(221)
    y, sr = librosa.load(filepath) # sr 為採樣頻率
    sound, _ = librosa.effects.trim(y)       # trim silent edges
    plt.title('machine sound')
    librosa.display.waveshow(y=sound, sr=sr) #.waveplot(sound, sr=sr)
    plt.ylabel('ampitude')
    
    plt.subplot(222)
    n_fft = 800
    D = np.abs(librosa.stft(sound[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    plt.title("FFT for sound")
    plt.plot(D)
    plt.xlabel('Hz')
    plt.ylabel('ampitude')
    
    plt.subplot(223)
    mel_spect=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=512,n_mels=64,power=2.0)
    librosa.display.specshow(librosa.power_to_db(mel_spect),sr=sr ,y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()

    ########################################################################414
    plt.subplot(224)
    plt.title('error score')
    plt.plot(error)
    plt.savefig("{result}/fig_{machine_type}_{machine_id}_{db}_{aorn}_{filename}.png".format(result=param["result_directory"],
                                            machine_type=machine_type,
                                            machine_id=machine_id,
                                            db=db,
                                            aorn=aorn,
                                            filename=filename))
def reset():
    testresult_en.delete(0 ,'end')
def show():
    global img
    global tk_img
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    db = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(filepath)[0])[0])[0])[0])[1]
    machine_type = os.path.split(os.path.split(os.path.split(os.path.split(filepath)[0])[0])[0])[1]
    machine_id = os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1]
    aorn = os.path.split(os.path.split(filepath)[0])[1]
    filename = os.path.split(filepath)[1]
    img = Image.open("{result}/fig_{machine_type}_{machine_id}_{db}_{aorn}_{filename}.png".format(result=param["result_directory"],
                                            machine_type=machine_type,
                                            machine_id=machine_id,
                                            db=db,
                                            aorn=aorn,
                                            filename=filename))
    tk_img = ImageTk.PhotoImage(img)
    imglabel = tk.Label(window, image=tk_img)
    imglabel.place(x=0,y=200)

####################################################################
#                           GUI                                    #
####################################################################
img = []
tk_img = []
window = tk.Tk()
window.title('TEST MIMII')
window.geometry('1500x800')
window.resizable(True, True)
'''Label'''
lb0 = tk.Label(text="Open wave File",bg ="grey",fg="white",height=1,width=15)
lb0.place(x=0 ,y=0)
lb1 = tk.Label(text="Open model File",bg ="grey",fg="white",height=1,width=15)
lb1.place(x=0 ,y=30)
lb4 = tk.Label(text="Test result",fg="black",height=1)
lb4.place(x=0 ,y=100)
'''Entry'''
loadFile_en = tk.Entry(width=100)
loadFile_en.place(x=110 ,y=0)
loadModel_en = tk.Entry(width=100)
loadModel_en.place(x=110 ,y=30)
testresult_en = tk.Entry(width=20)
testresult_en.place(x=110 ,y=100)
'''Button'''
loadFile_btn = tk.Button(text="...",height=1,command=loadFile)
loadFile_btn.place(x= 810,y=0)
loadModel_btn = tk.Button(text="...",height=1,command=loadModel)
loadModel_btn.place(x=810 ,y=30)
starteval_btn = tk.Button(text="start testing",height=1,command=eval)
starteval_btn.place(x=0 ,y=60)
clean_btn = tk.Button(window, text="reset",height=1,command=reset)
clean_btn.place(x=250 ,y=100)
img_btn = tk.Button(window, text="Visualization",height=1,command=show)
img_btn.place(x=0 ,y=140)

window.mainloop()
