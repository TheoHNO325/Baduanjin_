import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset,DataLoader
import random

class pth_Loader(Dataset):
    def __init__(self,pth,drop_ratio=0.5):
        self.scores = []
        self.student = []
        self.standard = []
        dataset_standard_path="/root/autodl-tmp/AlphaPose/dataset_processed14/standard_"
        self.t_pths=[]
        self.drop_ratio=drop_ratio
        for i in range(14):
            self.t_pths.append(torch.load(dataset_standard_path+str(i)+'.pth'))
        for data in self.t_pths:
            data[np.isnan(data).bool()]=0
        for student in pth:
            student['student'][np.isnan(student['student']).bool()]=0
            student['standard'][np.isnan(student['standard']).bool()]=0
            self.scores.append(student['score'])
            self.student.append(student['student'])
            self.standard.append(student['standard'])
        
    def __len__(self):
        return len(self.student)

    def __getitem__(self,index):
        if random.random()<self.drop_ratio:
            
            fake_data=self.t_pths[random.randint(0,13)]
            while torch.sum(torch.abs(fake_data-self.standard[index])) <= 1e-8:
                fake_data=self.t_pths[random.randint(0,13)]
            return self.student[index],self.t_pths[random.randint(0,13)],self.scores[index]*0
        else:
            return self.student[index],self.standard[index],self.scores[index]

if __name__=='__main__':
    dataset_path = "/root/autodl-tmp/AlphaPose/dataset_processed14/overall.pth"
    pth = torch.load(dataset_path)
    pth = pth_Loader(pth)