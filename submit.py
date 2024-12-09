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
from transformer import Model
import csv

root_path='/root/autodl-tmp/' # TODO
args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

# dataset_path = '/root/autodl-tmp/dataset/训练数据集'
# output_path = '/root/autodl-tmp/AlphaPose/processed'
# 直接终端输入 python video_demo.py (--save_video)即可

# 改:output_path = '/root/autodl-tmp/AlphaPose/dataset_processed

# dataset_path = args.datasetpath
dataset_path = '/root/autodl-tmp/test_video' # TODO

csv_path = '/home/service/result/18768423055/'
os.makedirs(csv_path,exist_ok=True)
output_path = args.outputpath 
score_None_max=20

subfolders = dataset_path
# subfolders.sort()
#['0-两手托天理三焦（八段锦）', '1-左右开弓似射雕（八段锦）', '10鹿抵（五禽戏）', '11鹿奔（五禽戏）', '12鸟伸（五禽戏）', '13鸟飞（五禽戏）', '14其他', '2-调理脾胃单臂举（八段锦）', '3-五劳七伤往后瞧（八段锦）', '4-摇头摆尾去心火（八段锦）', '5-两手攀足固肾腰（八段锦）', '6-攒拳怒目增气力（八段锦）', '7背后七颠百病消（八段锦）', '8虎举（五禽戏）', '9虎扑（五禽戏）', 'reference']
# subfolders=subfolders[:5]+subfolders[7:]

class FolderWrapper(Dataset):
    def __init__(self,subfolders):
        self.video_files=[]

        standard_folder = os.path.join(output_path, "standard")
        if not os.path.exists(standard_folder):
            os.makedirs(standard_folder)
        
        subfolder=subfolders
        folder_path = os.path.join(dataset_path, subfolder)
        output_subfolder = os.path.join(output_path, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        self.video_files.extend(video_files)
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self,index):
        return self.video_files[index]
    
standard_folder = os.path.join(output_path, "standard")

if __name__ == "__main__":
    # load model
    model = Model(input_size=42, frame_size=200, hidden_size=64, head_num=8, num_layers=12).cuda()
    model.load_state_dict(torch.load(root_path+'AlphaPose/model.pth'))
    model.eval()

    # load standards
    standards_path=root_path + "AlphaPose/standard/standard_"
    standards=[]
    attention_mask=torch.zeros([1,200,200]).cuda().bool()
    for i in range(14):
        standard_i_path=standards_path+str(i)+".pth"
        data=torch.load(standard_i_path).cuda().unsqueeze(0)
        data[np.isnan(data.cpu()).bool()]=0
        standards.append(data)

    folderset=FolderWrapper(subfolders)
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset,model_path=root_path+'AlphaPose/models/sppe/duc_se.pth')
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    print('Loading YOLO model..')
    det_model = Darknet(root_path+"AlphaPose/yolo/cfg/yolov3-spp.cfg")
    det_model.load_weights(root_path+'AlphaPose/models/yolo/yolov3-spp.weights')
    det_model.net_info['height'] = opt.inp_dim
    det_inp_dim = int(det_model.net_info['height'])

    pth_result = []
    csv_labels=[]
    csv_filenames=[]
    csv_scores=[]
    csv_using_times=[]
    # for v_i in range(14,len(folderset)):
    for v_i in range(len(folderset)):
        videofile = folderset[v_i]
        start_time_one_video = time.time()
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

        frame_interval = int(fps / target_fps)
        batchSize = args.batchsize
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        writer = pthWriter(args.save_video,fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize=frameSize)
        
        real_batches = int(num_batches/frame_interval)

        im_names_desc =  tqdm(range(real_batches),leave=False)
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
            if blank >= score_None_max:
                print("No result")
                break
            for k in range(i*batchSize, min((i + 1)*batchSize, datalen)):
                
                for _ in range(frame_interval - 1):  
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
                     #det_model返回了[1,22743,85] 22743为检测框的总数,85为(4边界框坐标+1置信度+80类别数(搞那么多类别干什么()))

                    dets = dynamic_write_results(prediction, opt.confidence,
                                        opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                    # if isinstance(dets, int) or dets.shape[0] == 0:
                    #     for k in range(len(orig_img)):
                    #         if Q.full():
                    #             time.sleep(2)
                    #         Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    #     continue
                    
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
                    # det_loader = DetectionLoader(data_loader,det_model, batchSize=args.detbatch).start()#start
                    # print(det_loader.datalen)
                    # det_processor = DetectionProcessor(det_loader).start() #start
                    # Data writer
                    with torch.no_grad():
                        # (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read() #DetectionProcessor,return Q.get()
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
                        # Pose Estimation
                        
                        # datalen = inps.size(0)
                        # leftover = 0
                        # if (datalen) % batchSize:
                        #     leftover = 1
                        # num_batches = datalen // batchSize + leftover
                        # hm = []
                        # for j in range(num_batches):
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
                        # TQDM
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
        final_result = writer.results().cuda().unsqueeze(0)
        final_result[torch.isnan(final_result)]=0
        scores=[]
        for i in range(14):
            scores.append(model(final_result,standards[i],attention_mask).item())
        label=torch.argmax(torch.Tensor(scores)).item()
        score=scores[label]
        file_name=videofile.split('/')[-1]
        using_time=int((time.time()-start_time_one_video)*1000)
        if score < 0.15: # 其它
            score=0
            label=14
        print(label,score,file_name,using_time)
        csv_labels.append(label)
        csv_scores.append(score)
        csv_filenames.append(file_name)
        csv_using_times.append(using_time)
        torch.cuda.empty_cache()
    csv_file = csv_path+"18768423055_submit.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i in range(len(csv_filenames)):
            writer.writerow([csv_filenames[i], csv_labels[i], csv_scores[i], csv_using_times[i]])
