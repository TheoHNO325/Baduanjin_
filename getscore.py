import json

annot = "/root/autodl-tmp/AlphaPose/video_annotations.json"

with open(annot,'r',encoding='utf-8') as file:
    data = json.load(file)

score_info = []

res = {}

for video in data:
    
    name = video["videoId"]
    if video["standardSocre"]:
        score = float(video["standardSocre"])
    else:score = None
    res[name] = score
    score_info.append(res)

print(res)