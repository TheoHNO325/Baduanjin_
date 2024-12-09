import json
import glob
import os

standard_file = glob.glob("/standard/*") #把所有视频的standard全部复制到一个下面
output_file = "/root/autodl-tmp/AlphaPose/standard_key"

with open('keyframes.json','r',encoding='utf-8') as file:
    keyframes = json.load(file)


for file in standard_file:
    file_name = os.path.basename(file)
    filename = os.path.splitext(file_name)[0]
    try:
        timestamp = keyframes[filename]
    except:
        raise "The file is not a standard file!"
    frame = []
    with open(file,'r',encoding='utf-8') as file:
        info = json.load(file)
    key_result = []
    for time in timestamp:
        time_parts = time.split(" ")[0].split(":")
        seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
        frame_index = 5*seconds # fps = 5
        key_result.append(info[frame_index])
    file_path = os.path.join(output_file, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(key_result, f)


    