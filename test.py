import numpy as np
from reconstructed import Frame
from reconstructed import Video,TeacherVideo
import json
import os

folder_path = "/root/autodl-tmp/AlphaPose/processed/0-两手托天理三焦（八段锦）"

teacher_path = "/root/autodl-tmp/AlphaPose/standard_key/standard_0.json"


for filename in os.listdir(folder_path):
        # 只处理.json后缀的文件
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)  # 获取完整的文件路径
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        with open(teacher_path, 'r', encoding='utf-8') as file:
            teacher_data = json.load(file)
        video = Video(data)
        # print(video.frames)
        teacher_Video = TeacherVideo(teacher_data)
        
        # print(teacher_Video.get_static_ranges(strict_threshold=1.5, relaxed_threshold=2.8))

        score = video.score_final(teacher_video=teacher_Video)

        print(filename,score)

        