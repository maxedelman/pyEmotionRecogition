from __future__ import print_function
import csv
from pydub import AudioSegment
import os
import glob
import random
import shutil

def m4aToMp3():
    path = '/Users/MaxEdelman/Documents/scienceResearchData/'

    for filename in os.listdir ( path ):
        name = str (filename)
        endlen = len (filename) - 3
        mp3Name = name[:endlen] + "mp3"
        AudioSegment.from_file (path+filename, format="m4a").export(path+mp3Name, format="mp3")


def m4aToMp3Glob():
    path = '/Users/MaxEdelman/Documents/scienceResearchm4aData/'

    for filename in glob.glob(os.path.join(path, '*.m4a')):
        name = str (filename)
        endlen = len (filename) - 3
        mp3Name = name[:endlen] + "mp3"
        AudioSegment.from_file (filename, format="m4a").export(mp3Name, format="mp3")

def m4aToMp3Glob_oneFile(filename):
    #m4a to mp3 for 1 file
    name = str (filename)
    endlen = len (filename) - 3
    mp3Name = name[:endlen] + "mp3"
    AudioSegment.from_file (filename, format="m4a").export(mp3Name, format="mp3")


def createTrainingDataList():
    destination_path = '/Users/MaxEdelman/Documents/pyEmotionRecognition/venv/code/'
    origin_path = '/Users/MaxEdelman/Documents/scienceResearchData/'
    trainingDataList_dict = {}

    for filename in glob.glob(os.path.join(origin_path,'*.wav')):
        name = str ( filename )
        key = name.split('Data/')
        keyName = key[1]
        # build mp3 name as name until the last underscore in the file name
        list = filename.split ('_')
        file_list = list[1].split ('.')
        emotion = file_list[0]
        # if the emotion is == 'Angry' then assign 1 to emotionPointer. Assign 1, 2, 3, to happy, nervous, sad respectively
        if emotion == 'Angry':
            emotionPointer = 1
        elif emotion == 'Happy':
            emotionPointer = 2
        elif emotion == 'Nervous':
            emotionPointer = 3
        elif emotion == 'Sad':
            emotionPointer = 4
        trainingDataList_dict[keyName] = emotionPointer

        with open ( destination_path + 'TrainingDataList.csv', 'w' ) as f:
            [f.write ( '{0},{1}\n'.format ( key, value ) ) for key, value in trainingDataList_dict.items()]


def mp3ToWAVGlob():
    path = '/Users/MaxEdelman/Documents/scienceResearchm4aData/'

    for filename in glob.glob(os.path.join(path, '*.mp3')):
        name = str (filename)
        endlen = len (filename) - 3
        WAVName = name[:endlen] + "wav"
        AudioSegment.from_file(filename, format="mp3").export(WAVName, format="wav")


def mp3ToWAVGlob_oneFile(filename):
    #mp3 to .wav for 1 file
    name = str ( filename )
    endlen = len ( filename ) - 3
    WAVName = name[:endlen] + "wav"
    AudioSegment.from_file ( filename, format="mp3" ).export ( WAVName, format="wav" )

def createTestingSet():
    path = "/Users/MaxEdelman/Documents/scienceResearchData/"
    destinationPath = '/Users/MaxEdelman/Documents/testingSet/'
    files = os.listdir(path)
    testingLength = len([name for name in os.listdir(destinationPath) if os.path.isfile(os.path.join(destinationPath, name))])
    for file in path:
        #print(testingLength)
        index = random.randrange(0, len(files))
        print (files[index])
        testingLength = testingLength + 1
        shutil.move(path + files[index], destinationPath + files[index])
        if testingLength == 20:
            break
def m4aToWAV():
    path = '/Users/MaxEdelman/Documents/scienceResearchm4aData/'
    destinationPath = '/Users/MaxEdelman/Documents/scienceResearchData/'
    junkPath = '/Users/MaxEdelman/Documents/JunkData/'
    for filename in glob.glob(os.path.join(path, '*.m4a')):
        name = str(filename)
        endlen = len ( filename ) - 3
        WAVName = name[:endlen] + "wav"
        extension = WAVName.split("/")[5]
        AudioSegment.from_file(filename, format="m4a").export(WAVName, format="wav")
        shutil.move(WAVName, destinationPath + extension)
        shutil.move(name, junkPath + extension)


#m4aToWAV()