#
#   Headpose Detection
#   Modified by Qhan
#   Last Update: 2019.1.9
#

import argparse
import cv2
import dlib
import numpy as np
import os
import os.path as osp
import pandas as pd

from timer import Timer
from utils import Annotator


t = Timer()

class HeadposeDetection():

    # 3D facial model coordinates
    landmarks_3d_list = [
        np.array([
            [ 0.000,  0.000,   0.000],    # Nose tip
            [ 0.000, -8.250,  -1.625],    # Chin
            [-5.625,  4.250,  -3.375],    # Left eye left corner
            [ 5.625,  4.250,  -3.375],    # Right eye right corner
            [-3.750, -3.750,  -3.125],    # Left Mouth corner
            [ 3.750, -3.750,  -3.125]     # Right mouth corner 
        ], dtype=np.double),
        np.array([
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
            [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
            [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
            [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
            [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
            [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
            [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
            [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
            [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
            [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
        ], dtype=np.double),
        np.array([
            [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
        ], dtype=np.double)
    ]

    # 2d facial landmark list
    lm_2d_index_list = [
        [30, 8, 36, 45, 48, 54],
        [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
        [33, 36, 39, 42, 45] # 5 points
    ]
    
    
    head_move_x_list = [
        -0.05768933615863209, 0.4624817647112173, -0.41989617786315836, -1.3942402546150017,
        -2.622525688576741, -2.864574859018932, -3.183193933107425, -4.3115627751028285,
        -5.930371167632655, -7.272487268011647, -7.961363213601079, -8.664985030093296,
        -8.549039184187412, -8.619535303531894, -6.615241827021224, -5.235016740980941,
        -3.950923563842068, -3.920389835550417, -3.4765685709053993, -3.2610399702035697,
        -3.371107797873432, -2.4814013914354347, -0.5211529322627431, 1.191575230425789,
        1.2215795497566504, 0.692075267300992, -0.6060450764064008, -2.1464077498539846,
        -3.5380441213412968, -4.210041133981368, -4.359937729789175, -5.630314792620588,
        -6.878548187341292, -8.426184756317982, -8.606534086291001, -7.917052015833634,
        -6.92932378063517, -6.228060133580227, -6.068434974820051, -5.109462434799199,
        -4.146164060254387, -2.9784994241413996, -2.6065841683320725, -1.3411528731001683,
        -0.08291347028406204   
    ]
    
    
    head_move_y_list = [
        -10.42173249679414, -12.439688655699586, -14.985145453751677, -17.75673993970492,
        -20.201356353872615, -18.77534482720391, -17.119838097677828, -13.895136615876524,
        -9.095153383787776, -2.468875118586527, 2.9046842765055456, 5.842905631916588,
        7.839929113887245, 10.39987714635792, 13.76329921463863, 16.60191775055887,
        18.880069969272398, 19.55244648781394, 19.491093850110577, 18.234227641273467,
        14.24919826784064, 9.661341886466733, 6.267347108452546, 4.491080092031438,
        2.234132929469296, -0.6644194494123566, -5.293084015852091, -8.918979089042116,
        -11.90000162057785, -13.207000855771607, -14.577298119503, -15.389240619195457,
        -14.599356029407405, -13.302216884857794, -10.381185246394336, -7.143830802859435,
        -2.3277699210243514, 1.1621729235455986, 4.962388322075049, 7.180009332452449,
        9.877786439665794, 12.494929995772347, 15.1138395811587, 16.816394494305445,
        17.70715419494105 
    ]

    def __init__(self, lm_type=1, predictor="model/shape_predictor_68_face_landmarks.dat", verbose=True):
        self.bbox_detector = dlib.get_frontal_face_detector()        
        self.landmark_predictor = dlib.shape_predictor(predictor)

        self.lm_2d_index = self.lm_2d_index_list[lm_type]
        self.landmarks_3d = self.landmarks_3d_list[lm_type]

        self.v = verbose
        
        self.flag = 0
        
        self.concent = 100
        
        self.x_list = []
        self.y_list = []


    def to_numpy(self, landmarks):
        coords = []
        for i in self.lm_2d_index:
            coords += [[landmarks.part(i).x, landmarks.part(i).y]]
        return np.array(coords).astype(np.int)

    def get_landmarks(self, im):
        # Detect bounding boxes of faces
        t.tic('bb')
        
        # im이 있으면 앞에 것 없으면 뒤에것 실행
        # get_frontal_face_detector()의 첫번째 인자는 이미지 벡터,
        # 두번째는 upsample_num_times이다.
        rects = self.bbox_detector(im, 0) if im is not None else []
            
        if self.v: 
            print(', bb: %.2f' % t.toc('bb'), end='ms')

        if len(rects) > 0:
            # Detect landmark of first face
            t.tic('lm')
            # 얼굴 부분만 자른 것이다.
            landmarks_2d = self.landmark_predictor(im, rects[0])

            # Choose specific landmarks corresponding to 3D facial model
            # 얼굴이미지에 landmark를 찍힌 좌표가 들어온다.
            landmarks_2d = self.to_numpy(landmarks_2d)
            if self.v: 
                print(', lm: %.2f' % t.toc('lm'), end='ms')
            
            # 얼굴의 상하좌우 좌표
            rect = [rects[0].left(), rects[0].top(), rects[0].right(), rects[0].bottom()]

            return landmarks_2d.astype(np.double), rect

        else:
            return None, None


    def get_headpose(self, im, landmarks_2d, verbose=False):
        h, w, c = im.shape
        f = w # column size = x axis length (focal length)
        u0, v0 = w / 2, h / 2 # center of image plane
        camera_matrix = np.array(
            [[f, 0, u0],
             [0, f, v0],
             [0, 0, 1]], dtype = np.double
         )
         
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1)) 

        # Find rotation, translation
        # 3차원 점에 관련된 카메라의 계산된 포즈를 리턴함.
        # tvec: xyz와 관련, rvec: 카메라의 방향 회전 벡터
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)
        
        if verbose:
            print("Camera Matrix:\n {0}".format(camera_matrix))
            print("Distortion Coefficients:\n {0}".format(dist_coeffs))
            print("Rotation Vector:\n {0}".format(rotation_vector))
            print("Translation Vector:\n {0}".format(translation_vector))

        return rotation_vector, translation_vector, camera_matrix, dist_coeffs


    # rotation vector to euler angles
    def get_angles(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec)) # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return [rx, ry, rz]

    # moving average history
    # history 비교를 통해 머리의 움직임 방향을 파악할 수 있다.
    history = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}
    
    # 데이터 저장
    def add_history(self, values):
        for (key, value) in zip(self.history, values):
            self.history[key] += [value]
  
    # 과거 데이터 삭제
    def pop_history(self):
        for key in self.history:
            self.history[key].pop(0)
            
    def get_history_len(self):
        return len(self.history['lm'])
            
    def get_ma(self):
        res = []
        for key in self.history:
            res += [np.mean(self.history[key], axis=0)]
        return res
    
    
    # 집중도를 표시하는 부분
    def draw_concent(self, im):
        # yellow
        fontColor = (0, 255, 255)
        h, w, c = im.shape
        fs = ((h+w)/2)/500
        px, py = int(5*fs), int(25*fs)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(im, "Concentration: %d" %self. concent,(px,py),font,fontScale=fs,color=fontColor)

    
    # 집중도 평가하는 부분
    def cal_angle(self, im, angle):
        x,y,z = angle
        
        self.x_list.append(x)
        self.y_list.append(y)
        
        
        if len(self.x_list) > 45:
            del self.x_list[0]
            
            df = pd.DataFrame({"v1":self.x_list, "v2":self.head_move_x_list})
            corr = df.corr(method="pearson")
            corr = corr.iloc[0,1]
            if corr > 0.8:
                print("****agree*****", corr)
                
        if len(self.y_list) > 45:
            del self.y_list[0]
            
            df = pd.DataFrame({"v1":self.y_list, "v2":self.head_move_y_list})
            corr = df.corr(method="pearson")
            corr = corr.iloc[0,1]
            if corr > 0.8:
                print("****not agree***", corr)
                
                
                
        # 화면 밖을 보고 있는지 판단
        if y > 23 or y < -23:
            print("화면을 안보고 있음!!")
            pass
        
        #print(self.y_list)
            
            
            
       
        
    # return image and angles
    def process_image(self, im, draw=True, ma=3):
        # landmark Detection
        # grayscale 진행, 3차원을 1차원으로 줄여 계산속도 향상
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 landmark좌표와 얼굴 box좌표
        landmarks_2d, bbox = self.get_landmarks(im_gray)

        # if no face deteced, return original image
        # 좌표가 검출되지 않았으면 기존이미지 다시 return
        if landmarks_2d is None:
            # 머리가 2분이상 검출되지 않은 경우 잠자고 있다고 탐지
            if self.flag == 0:
                t.tic('sleep')
                self.flag=1
            else:
                if t.toc('sleep') > 120000:
                    print("sleep!!!")
            
            
            if self.concent > 20:
                self.concent -= 1
            # 집중도 표시
            self.draw_concent(im)
            return im, None
        
        if self.concent < 100:
            self.concent += 1
        self.flag = 0

        # Headpose Detection
        t.tic('hp')
        rvec, tvec, cm, dc = self.get_headpose(im, landmarks_2d)
        if self.v: 
            print(', hp: %.2f' % t.toc('hp'), end='ms')
            
        # 몇개나 기억할 것인지
        if ma > 1:
            self.add_history([landmarks_2d, bbox, rvec, tvec, cm, dc])
            if self.get_history_len() > ma:
                self.pop_history()
            landmarks_2d, bbox, rvec, tvec, cm, dc = self.get_ma()

        t.tic('ga')
        angles = self.get_angles(rvec, tvec)
        if self.v: 
            print(', ga: %.2f' % t.toc('ga'), end='ms')
        
        # 각도에 따른 집중도 및 이해도 판단 함수
        self.cal_angle(im, angles)
          
        
        if draw:
            t.tic('draw')
            annotator = Annotator(im, angles, bbox, landmarks_2d, rvec, tvec, cm, dc, b=10.0, concent = self.concent)
            im = annotator.draw_all()
            if self.v: 
                print(', draw: %.2f' % t.toc('draw'), end='ms' + ' ' * 10)
         
        return im, angles
    

    

# def main(args):
#     in_dir = args["input_dir"]
#     out_dir = args["output_dir"]

#     # Initialize head pose detection
#     hpd = HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

#     for filename in os.listdir(in_dir):
#         name, ext = osp.splitext(filename)
#         if ext in ['.jpg', '.png', '.gif']: 
#             print("> image:", filename, end='')
#             image = cv2.imread(in_dir + filename)
#             res, angles = hpd.process_image(image)
#             cv2.imwrite(out_dir + name + '_out.png', res)
#         else:
#             print("> skip:", filename, end='')
#         print('')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', metavar='DIR', dest='input_dir', default='images/')
#     parser.add_argument('-o', metavar='DIR', dest='output_dir', default='res/')
#     parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
#     parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
#                         default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
#     args = vars(parser.parse_args())

#     if not osp.exists(args["output_dir"]): os.mkdir(args["output_dir"])
#     if args["output_dir"][-1] != '/': args["output_dir"] += '/'
#     if args["input_dir"][-1] != '/': args["input_dir"] += '/'
#     main(args)