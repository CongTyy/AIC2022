import torch
import clip
from PIL import Image
import numpy as np
from glob import glob
import pathlib
from tqdm import tqdm
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

# for i1 in tqdm(range(100)):
#     for i2 in enumerate(tqdm(a, leave=False)):

device = "cuda:1"
model, preprocess = clip.load("ViT-B/16", device=device)
with torch.no_grad():
    video_path = sorted(glob('KeyFrames2/*'))
    for video_path in tqdm(video_path): #KeyFramesC00_V00
        for key_path in tqdm(sorted(glob(f'{video_path}/*')), leave=False): #C00_V0000
            for img_path in tqdm((sorted(glob(f'{key_path}/*.jpg'))), leave=False):
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                npy_name = pathlib.Path(img_path).parent.resolve() 
                # print(f'{npy_name}/{img_path.split("/")[-1][:-4]}.npy', image_features)
                np.save(f'{npy_name}/{img_path.split("/")[-1][:-4]}.npy', image_features.cpu().numpy())

        # csv_path = p_path + key_path.split('/')[-1]  + '.csv'
        # # print(csv_path)
        # IDs = {}
        # csvfile = open(csv_path,'rb')
        # timeReader = csv.reader(csvfile, delimiter = ',')
        # for row in timeReader:
        #     IDs[row[0]] = row[1]

        # for npy_path, (old, new) in zip(sorted(glob(f'{key_path}/*.npy')), IDs.items()):
        #     name_npy = npy_path.split("/")[-1][:-4]
        #     old = old[:-4]
        #     if name_npy == old:
        #         os.rename(npy_path, f'{npy_path[:-10]}{new}.npy')
        #         os.rename(f'{npy_path[:-4]}.jpg', f'{npy_path[:-10]}{new}.jpg')