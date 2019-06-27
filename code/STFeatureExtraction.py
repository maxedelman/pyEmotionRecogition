from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from openpyxl import workbook

def STFeatureExtraction():
    path = '/Users/MaxEdelman/Documents/scienceResearchData'
    #meanList = []
    meanDf = pd.DataFrame([])
    for file in os.listdir('/Users/MaxEdelman/Documents/scienceResearchData'):
        extension = os.path.splitext(file)[1]
        if extension == '.wav':
            #feature extraction
            (Fs, x) = audioBasicIO.readAudioFile('/Users/MaxEdelman/Documents/scienceResearchData/' + file)
            F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
            #making the variable F accessible to make into a dataframe
            F = np.transpose(F)
            F = F[0, :, :]
            F = np.transpose(F)
            #making initial dataframe
            features = pd.DataFrame(F, columns=f_names)
            mean = features.describe()
            #mean dataframe
            mean = mean.iloc[[1]]
            #designates each emotion to a number and stores it in the df
            if "Angry" in file:
                emotionPointer = 0
            elif "Happy" in file:
                emotionPointer = 1
            elif "Nervous" in file:
                emotionPointer = 2
            elif "Sad" in file:
                emotionPointer = 3
            mean['emotion'] = emotionPointer
            #creates 1 df of all of the means
            meanDf = meanDf.append(mean)
    meanDf.to_excel('/Users/MaxEdelman/Documents/meanDf.xlsx')


def fixEmotion():
    df = pd.read_excel('/Users/MaxEdelman/Documents/meanDf.xlsx')
    emotionColumn = df['emotion']
    print(emotionColumn)

fixEmotion()