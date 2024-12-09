import numpy as np
import matplotlib.pyplot as plt

class Frame:
    def __init__(self, dist):
        self.image_id = int(dist['image_id'].split('.')[0])
        self.point = -np.array(dist['keypoints']).reshape(17, 2)
        
        self.angle = dist['angle_keys']
        
    def __str__(self):
        output = "Image ID:" + str(self.image_id) +'\n' + "Angle:" + str(self.angle)
        return output

    def plot(self):
        plt.plot(self.point[:, 0], self.point[:, 1], 'o')
        plt.axis('equal')
        plt.show()
        
    def distance(self, other):
        sum = 0
        for i in range(17):
            sum += ((self.point[i][0]-other.point[i][0])**2 + (self.point[i][1]-other.point[i][1])**2)**0.5
        return sum/17
    
    def is_acting(self, other, threshold = 1.0):
        return self.distance(other) > threshold
    
    def score(self, teacher_frame) -> float:
        total_score = 0
        teacher_angles = teacher_frame.angle
       
        
        params = {
            'left_arm_bend': (0.1, [10, 20, 30], [1,0.8,0.6]),
            'right_arm_bend': (0.1, [10, 20, 30], [1,0.8,0.6]),
            'left_armpit': (0.25, [10, 30, 50], [1,0.8,0.6]),
            'right_armpit': (0.25, [10, 30, 50], [1,0.8,0.6]),
            'left_waist_leg': (0.1, [10, 20, 30], [1,0.8,0.6]),
            'right_waist_leg': (0.1, [10, 20, 30], [1,0.8,0.6]),
            'left_knee': (0.05, [10, 20, 30], [1,0.8,0.6]),
            'right_knee': (0.05, [10, 20, 30], [1,0.8,0.6]),
        }

        keys = params.keys()
        for key in keys:
            param = params[key]
            angle_diff = abs(self.angle[key] - teacher_angles[key])
            # if angle_diff > 0 and angle_diff < param[1][0]:
            #     total_score += param[0] * param[2][0]
            # elif angle_diff >= param[1][0] and angle_diff < param[1][1]:
            #     total_score += param[0] * param[2][1]
            # elif angle_diff >= param[1][1] and angle_diff < param[1][2]:
            #     total_score += param[0] * param[2][2]
            # print('current pos',key)
            # print('current score',total_score)
            
            total_score += param[0]*func_angle(angle_diff,key) #换成线性的会不会好一点
            # print(total_score)
        return total_score


class Video:
    def __init__(self, List):
        self.frames = [0] * (int(List[-1]['image_id'].split('.')[0]) + 1)  # 初始化帧列表
        
        for dist in List:
            self.frames[int(dist['image_id'].split('.')[0])] = Frame(dist)

    def __len__(self):
        return len(self.frames)
 
    def plot(self):
        for frame in self.frames:
            if frame.image_id%20 == 0:
                frame.plot()

    # [-300,1000]ms + fps=5  --> [-2,5] image id

    def score_final(self, teacher_video, alpha = 0.13):
        teacher_frames = teacher_video.frames
        
        standard_keyframes = teacher_video.standard_keyframes
        score_range = teacher_video.score_range

        highest_scores = []
        key_frames = []
        num = 0
        for each_range in score_range:
            max_score = 0.00
            key=each_range[0]
            getone = np.array([])
            for i in each_range:
                
                if self.frames[i] and self.frames[i-1]:
                    # print(self.frames[i-1])
                    if not self.frames[i].is_acting(self.frames[i-1]):
                        continue
                    else:
                        if teacher_frames[i] == 0:
                            continue
                        score = self.frames[i].score(teacher_frames[i])
                        if score >= max_score:
                            max_score = score
                            key = i
                            if score==1:
                                getone = np.append(getone,i)
                                
            if np.any(getone):
                closest_one = getone - standard_keyframes[num]
                key = getone[np.argmin(np.abs(closest_one))]        
                  
            key_frames.append(key)               
            highest_scores.append(max_score)
            num += 1

        standard_diff,weight = teacher_video.compute_diff()
        
        diff = np.diff(np.array(key_frames))
        
        frame_difference = np.sqrt(np.sum((standard_diff-diff)**2)) #最小二乘损失?
        
        frame_score = func(frame_difference) #关键帧之间的差值
        
        angle_score = np.sum(weight * np.array(highest_scores))

        return alpha*frame_score + (1-alpha)*angle_score  #试图这么干,不过这样一来标准视频给我干到0.9了--- 经过一番操作,又干回1.0了


class TeacherVideo(Video):
    def __init__(self,List):
        Video.__init__(self,List)
        self.standard_keyframes = np.array(self.frames).nonzero()[0]
        self.length = len(self)
        self.score_range = []

        for key in self.standard_keyframes:
            self.score_range.append(range(max(0,key-5),min(key+3,self.length)))
            for i in self.score_range[-1]:
                self.frames[i] = self.frames[key]
        
    def compute_diff(self):
        standard_diff = np.diff(self.standard_keyframes)
        weight = np.insert(standard_diff,0,self.standard_keyframes[0]).astype(float)
        weight /= float(self.standard_keyframes[-1])
        return standard_diff,weight


def func(x,k=0.8):             

    return 2 / (1 + np.exp(k * x))         
                    
            
def func_angle(x,mode):
    if x < 10:score =1
    else:
        if mode in ["left_arm_bend","right_arm_bend","left_waist_leg","right_waist_leg","left_knee","right_knee"]:
            score = 1-0.2*(x-10)
        elif mode in ["left_armpit","right_armpit"]:
            score = 1-0.1*(x-10)
    return score    