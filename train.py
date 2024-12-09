import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt
from torch.utils.data import Dataset
import json

from dataloader import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, pthWriter,Mscoco
from dataloader_2 import pth_Loader
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
from yolo.darknet import Darknet

import ntpath
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2
import glob
from pPose_nms import pose_nms, write_json,write_pth
from torch.utils.data import DataLoader

from yolo.preprocess import prep_image, prep_frame, inp_to_image
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from dataloader import crop_from_dets
from pPose_nms import pose_nms, write_json
from matching import candidate_reselect as matching
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from reconstructed import *
from transformer import *


# 得到视频标准评分
annot = "/root/autodl-tmp/AlphaPose/video_annotations.json"

with open(annot,'r',encoding='utf-8') as file:
    data = json.load(file)
score_info = {}

for video in data:
    name = video["videoName"]
    if video["standardSocre"]:
        score = float(video["standardSocre"])
    else:score = None
    score_info[name] = score

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


dataset_path = args.datasetpath
output_path = args.outputpath
score_None_max=10

if not os.path.exists(output_path):
    os.mkdir(output_path)

subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
subfolders.sort()
#['0-两手托天理三焦（八段锦）', '1-左右开弓似射雕（八段锦）', '10鹿抵（五禽戏）', '11鹿奔（五禽戏）', '12鸟伸（五禽戏）', '13鸟飞（五禽戏）', '14其他', '2-调理脾胃单臂举（八段锦）', '3-五劳七伤往后瞧（八段锦）', '4-摇头摆尾去心火（八段锦）', '5-两手攀足固肾腰（八段锦）', '6-攒拳怒目增气力（八段锦）', '7背后七颠百病消（八段锦）', '8虎举（五禽戏）', '9虎扑（五禽戏）', 'reference']


class FolderWrapper(Dataset):
    def __init__(self,subfolders):
        self.video_files=[]
        self.output_subfolders=[]
        self.standard_videos=[]
        self.standard_index = []

        standard_folder = os.path.join(output_path, "standard")
        
        for subfolder in subfolders:
            folder_path = os.path.join(dataset_path, subfolder)
            output_subfolder = os.path.join(output_path, subfolder)
            video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
            for video_file in video_files:
                # 如果文件以 "standard" 开头，移动到 standard_folder
                if os.path.basename(video_file).startswith("standard"):
                    now_standard = os.path.basename(video_file).split('.')[0]
                    self.standard_videos.append(video_file)
                    video_files.remove(video_file)
            self.video_files.extend(video_files)
            self.output_subfolders.extend([output_subfolder]*len(video_files))
            self.standard_index.extend([now_standard]*len(video_files)) 
            
        self.video_files[:0]= self.standard_videos

        self.output_subfolders[:0]=[standard_folder]*len(self.standard_videos)
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self,index):
        return self.video_files[index],self.output_subfolders[index],self.standard_index[index]
    
standard_folder = output_path

if __name__ == "__main__":

    folderset=FolderWrapper(subfolders)
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    print('Loading YOLO model..')
    det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights('models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = opt.inp_dim
    det_inp_dim = int(det_model.net_info['height'])

    pth_result = []

    # for v_i in range(14,len(folderset)):
    
    for v_i in range(14,20):
        videofile,output_subfolder,standard_index = folderset[v_i]
        print(standard_index)
        video_score = score_info[os.path.basename(videofile)]
        print(video_score)
        parent_dir = os.path.basename(os.path.dirname(videofile))
        print(parent_dir)
        print("videofile",videofile)
        print("output_subfolder",output_subfolder)

        mode = args.mode

        if not len(videofile):
            raise IOError('Error: must contain --video')
        
        stream = cv2.VideoCapture(videofile)
        assert stream.isOpened(), 'Cannot capture source'
        fourcc=int(stream.get(cv2.CAP_PROP_FOURCC))
        fps=stream.get(cv2.CAP_PROP_FPS)
        frameSize=(int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        target_fps = 5
        save_path = os.path.join(output_subfolder, 'AlphaPose_'+ntpath.basename(videofile).split('.')[0]+'.avi')

        frame_interval = int(fps / target_fps)
        batchSize = args.batchsize
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        
        save_path = os.path.join(output_subfolder, 'AlphaPose_'+ntpath.basename(videofile).split('.')[0]+'.avi')
        writer = pthWriter(args.save_video,save_path,cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize)
        
        real_batches = int(num_batches/frame_interval)
        print(real_batches)

        im_names_desc =  tqdm(range(real_batches),leave=False)
        print(im_names_desc)
        batchSize_useless = args.posebatch
        score_None_num=0
    
        det_model.cuda()
        det_model.eval()
        blank = 0 

        runtime_profile = {
                        'dt': [],
                        'pt': [],
                        'pn': []
                    }
        for i in im_names_desc:
            start_time = getTime()
            if blank >= 10:
                print("No result")
                break
            for k in range(i*batchSize, min((i + 1)*batchSize, datalen)):
                
                for _ in range(frame_interval - 1):  # 跳过多余的帧
                    stream.read()
                    
                inp_dim = int(opt.inp_dim)
                
                (grabbed, frame) = stream.read()
                if not grabbed:
                    break
                img_k, orig_img, im_dim_list = prep_frame(frame, inp_dim)
                im_name = str(k)+'.jpg'
                
                with torch.no_grad():
                # Human Detection
                    
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
                    im_dim_list_ = im_dim_list
                    
                    img = img_k.cuda()
                
                    prediction = det_model(img, CUDA=True)
                    # NMS process

                    dets = dynamic_write_results(prediction, opt.confidence,
                                        opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                    
                    if not torch.is_tensor(dets):
                        blank += 1
                        continue
                    dets = dets.cpu()
                    im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                    scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                    # coordinate transfer
                    dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                    dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                    
                    dets[:, 1:5] /= scaling_factor
                    for j in range(dets.shape[0]):
                        dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                        dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                    boxes = dets[:, 1:5]
                    scores = dets[:, 5:6]
                    
                    inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
                    pt1 = torch.zeros(boxes.size(0), 2)
                    pt2 = torch.zeros(boxes.size(0), 2)
                    inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                    inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                
                    sys.stdout.flush()
                    with torch.no_grad():
                        if scores is None:
                            score_None_num+=1
                        if score_None_num >= score_None_max:
                            break
                        if orig_img is None:
                            break
                        if boxes is None or boxes.nelement() == 0:
                            writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                            writer.update()
                            continue

                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)

                        inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                        hm = pose_model(inps_j)

                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)

                        hm = hm.cpu().data
                        writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                        writer.update()
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

                    if args.profile:
                        im_names_desc.set_description(
                        'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                        )
        

        print('===========================> Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
        st_time=time.time()
        while(writer.running()): # not Q.empty()
            if time.time()-st_time>3:
                print("Warning:Waiting for running end time out.")
                break
            # print("running...")
            pass
        final_result = writer.results()
        if v_i < 14:
            final_result = write_pth(final_result, output_subfolder,os.path.basename(videofile).split('.')[0])
        else:
            standard_result = torch.load(os.path.join(standard_folder,standard_index+".pth"))
            final_result = {
                "score":video_score,
                "student":final_result,
                "teacher":standard_result
            }
            pth_result.append(final_result)

        torch.cuda.empty_cache()

    pth_result = write_pth(pth_result, output_path,"overall_result")



