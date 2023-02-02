import sys, re, math, time, numpy as np, matplotlib.pyplot as plt, json, pickle, collections, pandas as pd
from collections import OrderedDict
from matplotlib.pyplot import cm

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,"Z": 25}
CHARPROTLEN = 25

CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60, "t": 61, "y": 62}
CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64

def orderdict_list(dict):
    x = []
    for d in dict.keys():
        x.append(dict[d])
    return x

def one_hot_smiles(line, maxsqLen, smilecheLen):
    X = np.zeros((maxsqLen, len(smilecheLen)))  # +1
    for i, ch in enumerate(line[:maxsqLen]):
        X[i, (smilecheLen[ch] - 1)] = 1
    return X  

def one_hot_sequence(line, maxseqLen, smilecheLen):
    X = np.zeros((maxseqLen, len(smilecheLen)))
    for i, ch in enumerate(line[:maxseqLen]):
        X[i, (smilecheLen[ch]) - 1] = 1

    return X  

def label_smiles(line, maxsqLen, smilecheLen):
    X = np.zeros(maxsqLen)
    for i, ch in enumerate(line[:maxsqLen]):  
        X[i] = smilecheLen[ch]

    return X  

def label_sequence(line, maxseqLen, smilecheLen):
    X = np.zeros(maxseqLen)
    for i, ch in enumerate(line[:maxseqLen]):
        X[i] = smilecheLen[ch]

    return X  

def get_removelist(list_name, length):
    removelist = []
    for i, x in enumerate(list_name):
        if len(x) >= length:
            removelist.append(i)
    return removelist

def list_remove(list_name, removelist):
    a_index = [i for i in range(len(list_name))]
    a_index = set(a_index)
    b_index = set(removelist)
    index = list(a_index - b_index)
    a = [list_name[i] for i in index]
    return a

def df_remove(dataframe, removelist, axis):
    if axis == 0:
        newDataframe = dataframe.drop(removelist)
        newDataframe = newDataframe.reset_index(drop=True)
    if axis == 1:
        newDataframe = dataframe.drop(removelist, axis=1)
        newDataframe.columns = range(newDataframe.shape[1])
    return newDataframe