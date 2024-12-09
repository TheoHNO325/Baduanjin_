import json
from torch.utils.data import Dataset
import os
import glob
import torch
import numpy as np


output_path = "/root/autodl-tmp/AlphaPose/dataset_processed14"
json_folder = "/root/autodl-tmp/AlphaPose/processed"

subfolders = [f for f in os.listdir(json_folder) if os.path.isdir(os.path.join(json_folder, f))]

class FolderWrapper(Dataset):
    def __init__(self,subfolders):
        self.json_files=[]
        self.output_subfolders=[]
        self.standard_jsons=[]
        self.standard_index = []

        standard_folder = os.path.join(output_path, "standard")
        if not os.path.exists(standard_folder):
            os.makedirs(standard_folder)
        
        standard_list = ["standard_0","standard_1","standard_2","standard_3","standard_4","standard_5","standard_6","standard_7","standard_8",
                         "standard_9","standard_10","standard_11","standard_12","standard_13"]

        for subfolder in subfolders:
            folder_path = os.path.join(json_folder, subfolder)
            now_standard = None
            json_files = glob.glob(os.path.join(folder_path, "*json"))
 
            for json_file in json_files:
                # 如果文件以 "standard" 开头，移动到 standard_folder
                if os.path.basename(json_file).startswith("standard"):
                    now_standard = os.path.basename(json_file).split('.')[0]
                    self.standard_jsons.append(json_file)
                    json_files.remove(json_file)

            self.json_files.extend(json_files)
            if now_standard:
                self.standard_index.extend([now_standard]*len(json_files)) 
            else:
                x = np.random.randint(0,14,size=(1,len(json_files)))
                for i in range(len(json_files)):
                    self.standard_index.append(standard_list[x[0][i]]) 
        
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self,index):
        return self.json_files[index],self.standard_index[index]

def getangles(keypoints):
    x = 0
    point = []

    for i in range(len(keypoints)):
        if i % 2 ==1:
            point.append([x,keypoints[i]])
        else:
            x = keypoints[i]
    angle = {}
    angle_list = []
    to_get = [[8,6,12],[10,8,6],[11,5,7],[9,7,5],[6,12,14],[13,11,5],[16,14,12],[11,13,15]]
    position_names = ['left_armpit','left_arm_bend','right_armpit','right_arm_bend','left_waist_leg','right_waist_leg','left_knee','right_knee']
    #->[[8,6,12],[10,8,6],[11,5,7],[9,7,5],[6,12,14],[13,11,5],[16,14,12],[11,13,15]]
    for num,posname in zip(to_get,position_names):
        a = point[int(num[0])]
        b = point[int(num[1])]
        c = point[int(num[2])]
        res = calculate_angle(a,b,c)
        angle[posname] = res
        angle_list.append(res)
    return angle,angle_list

def calculate_angle(A, B, C,turn_deg = True):
    
    eps = 10e-10
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)
    
    BA_norm = np.linalg.norm(BA)
    BC_norm = np.linalg.norm(BC)
    
    dot_product = np.dot(BA, BC)
    
    angle_rad = np.arccos(dot_product / (BA_norm * BC_norm+eps))
    
    if turn_deg:
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    else:
        return angle_rad
    
def write_pth_for_standard(json_data,filename):
    final_res = torch.zeros(200,42)
    frames_length = len(json_data)
    if frames_length <= 200:
        for i in range(frames_length):
            keypoints = torch.tensor(json_data[i]["keypoints"])
            _,angles = getangles(keypoints)
            angles = torch.tensor(angles)
            keypoints = torch.cat((keypoints,angles))
            final_res[i] = keypoints
    else:
        for i in range(0,200):
            keypoints = torch.tensor(json_data[i]["keypoints"])
            _,angles = getangles(keypoints)
            angles = torch.tensor(angles)
            keypoints = torch.cat((keypoints,angles))
            final_res[i] = keypoints
    path = os.path.join(output_path,filename+'.pth')
    torch.save(final_res,path)
    return final_res


def get_pth(json_data,score,standard_name):

    final_res = torch.zeros(200,42)
    frames_length = len(json_data)
    if frames_length <= 200:
        for i in range(frames_length):
            keypoints = torch.tensor(json_data[i]["keypoints"])
            _,angles = getangles(keypoints)
            angles = torch.tensor(angles)
            keypoints = torch.cat((keypoints,angles))
            final_res[i] = keypoints
    else:
        for i in range(0,min(frames_length,200)):
            keypoints = torch.tensor(json_data[i]["keypoints"])
            _,angles = getangles(keypoints)
            angles = torch.tensor(angles)
            keypoints = torch.cat((keypoints,angles))
            final_res[i-5] = keypoints

    standard_file = os.path.join(output_path,standard_name)
    st = torch.load(standard_file)
    print(st.shape,standard_file)
    final_res = {
        "score":score,
        "student":final_res,
        "standard":st
    }
    return final_res

folder = FolderWrapper(subfolders)

st = folder.standard_jsons

#得到评分
annot = "/root/autodl-tmp/AlphaPose/video_annotations.json"

with open(annot,'r',encoding='utf-8') as file:
    data = json.load(file)
    
score_info = {}
from tqdm import tqdm
for video in tqdm(data):
    name = video["videoId"]
    if video["standardSocre"]:
        score = float(video["standardSocre"])
    else:score = 0
    score_info[name] = score
###

for i in tqdm(range(len(st))):
    st_files = st[i]
    with open(st_files,'r',encoding='utf-8') as file:
        data = json.load(file)
    filename = os.path.basename(st_files).split(".")[0]
    final_res = write_pth_for_standard(data,filename)


res = []

for i in tqdm(range(len(folder))):
    json_file,standard_file = folder[i]

    filename = os.path.basename(json_file).split(".")[0]
    the_score = score_info[filename]
    standard_name = standard_file+".pth"
    with open(json_file,'r',encoding='utf-8') as file:
        data = json.load(file)

    res.append(get_pth(data,the_score,standard_name))
    
torch.save(res,output_path+"/overall.pth")



