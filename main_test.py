import shutil
import clip
import torch
import numpy as np
from glob import glob
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import norm 
from sklearn.metrics.pairwise import cosine_distances

import torchtext
from torchtext.data import get_tokenizer

class Main:
    def __init__(self, 
                keyframe_path = "KeyFrames", 
                feature_path = 'CLIPFeatures') -> None:
        '''
        dict{
            path:
            index:
            embedded:
        }
        '''
        # self.feature_path = sorted(glob(f'{feature_path}/*.npy'))
        self.feature_path = sorted(glob(f'{feature_path}/*'))
        self.video_path = sorted(glob(f'{keyframe_path}/*'))
        self.dataset_dict = []
        self.embs = []
        self.device = 'cuda:1'
        # Load Clip
        #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
        self.model, preprocess = clip.load("ViT-B/16", device=self.device)
        self.model.eval()

        self.output = {}
        

    def extract_npy(self):
        for k, f_path in enumerate(self.feature_path):
            vd_path = sorted(glob(f'{self.video_path[k]}/*'))
            for i, npy in enumerate(sorted(glob(f'{f_path}/*.npy'))):
                npy_file = np.load(npy)
                for j, f_name in enumerate(sorted(glob(f'{vd_path[i]}/*.jpg'))):
                    name = f'{f_name.replace("jpg", "npy")}'
                    np.save(name, npy_file[j])

    def create_dict(self):
        embs = []
        dataset_dict = []
        i = 0
        for video_path in self.video_path: #KeyFramesC00_V00
            for key_path in sorted(glob(f'{video_path}/*')): #C00_V0000
                for npy_path in sorted(glob(f'{key_path}/*.npy')):
                    npy = np.load(npy_path)
                    # npy /= norm(npy, axis=-1, keepdims=True)
                    npy = npy.reshape(1,512).T
                    embs.append(npy)
                    dataset_dict.append({
                        'path': npy_path.replace('npy', 'jpg'),
                        'index': i,
                        'embedded': npy
                    })
                    i += 1
        embs = np.hstack(embs)
        embs = torch.tensor(embs, dtype=torch.float32).to(self.device)
        return embs, dataset_dict

    def pre_processing(self):
        # self.extract_npy()
        embs, dataset_dict = self.create_dict()
        return embs, dataset_dict
    def create_workspace(self):
        if os.path.exists('static'):
            shutil.rmtree('static')
        
        os.mkdir('static')
    

    def inference(self, embs, dataset_dict):

        '''
        text_features: 1 512
        embs: 512 N
        '''

        # self.pre_processing()
        similarity = 0
        index = 0
        query_name = input("query_number(1,2,3,..): ")
        query_name = "query-"+ query_name + '.csv'
        text_search = input("text_search: ")
        num_img = input("num_img: ")

        tokenizer = get_tokenizer("basic_english")
        tokens = tokenizer(text_search)
        print(tokens)
        text_tokens = clip.tokenize(tokens) # pre-encoder
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens.to(self.device)).to(torch.float32)# 1 512
            

        # embs /= embs.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = list(((text_features @ embs).cpu().numpy()).reshape((len(dataset_dict))))

        # similarity = list((((text_features @ embs).softmax(dim=-1)).cpu().numpy()).reshape((len(dataset_dict))))
        print(max(similarity))
        index = list(np.argsort(similarity))
        index.reverse()
        for i in range(0,5):
            print(dataset_dict[index[i]]['path'])

        
        # print(similarity[index[0]])
        # print(time.time() - t1)

        # #---save_csv----
        idfram = []
        idvideo = []
        idpath = []
        self.create_workspace()
        for i in range(0,int(num_img)):
            video_name = dataset_dict[index[i]]['path'].split('/')[-2]
            frame_name = dataset_dict[index[i]]['path'].split('/')[-1][:-4]
            shutil.copy(dataset_dict[index[i]]['path'], f"static/{dataset_dict[index[i]]['path'].split('/')[-1]}")
            idfram.append(frame_name)
            idvideo.append(video_name + ".mp4")
            idpath.append(f"{dataset_dict[index[i]]['path'].split('/')[-1]}")
        # np.savetxt(query_name, [p for p in zip(idvideo, idfram)], delimiter=',', fmt='%s')

        output = {
            'video_name': idvideo,
            'frame_name': idfram,
            'paths': idpath
        }
        return output, query_name

if __name__=="__main__":
    main = Main()
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer('The artist is standing on high scaffolding and restoring paintings on the wall. On the wall are portraits of 5 male characters. Background of yellow and red painting. In the picture there is also a person riding a horse.')
    text_tokens = clip.tokenize(tokens)
    print(tokens)
    # main.pre_processing()