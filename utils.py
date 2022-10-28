import hnswlib
import numpy as np
import pickle
import torch
import time
# import pandas as
import csv 
from glob import glob
import os, unicodecsv as csv
# open and store the csv file




# with open('C00_V0000.csv','rb') as csvfile:
#     timeReader = csv.reader(csvfile, delimiter = ',')
#     # build dictionary with associated IDs
#     for row in timeReader:
#         IDs[row[0]] = row[1]
# print(IDs)
p_path = "P/keyframe_p/"

video_path = sorted(glob('KeyFrames/*'))
for video_path in video_path: #KeyFramesC00_V00
    for key_path in sorted(glob(f'{video_path}/*')): #C00_V0000
        
        csv_path = p_path + key_path.split('/')[-1]  + '.csv'
        # print(csv_path)
        IDs = {}
        csvfile = open(csv_path,'rb')
        timeReader = csv.reader(csvfile, delimiter = ',')
        for row in timeReader:
            IDs[row[0]] = row[1]

        for npy_path, (old, new) in zip(sorted(glob(f'{key_path}/*.npy')), IDs.items()):
            name_npy = npy_path.split("/")[-1][:-4]
            old = old[:-4]
            if name_npy == old:
                os.rename(npy_path, f'{npy_path[:-10]}{new}.npy')
                os.rename(f'{npy_path[:-4]}.jpg', f'{npy_path[:-10]}{new}.jpg')
            
# move files
# path = '/home/hoangtv/Desktop/Ty/AIC/C00_V0000/'
# tmpPath = '/home/hoangtv/Desktop/Ty/AIC/new/'
# for oldname in os.listdir(path):
#     # ignore files in path which aren't in the csv file
#     if oldname in IDs:
#         try:
#             os.rename(os.path.join(path, oldname), os.path.join(tmpPath, IDs[oldname] + ".jpg"))
#         except:
#             print ('File ' + oldname + ' could not be renamed to ' + IDs[oldname] + '!')

