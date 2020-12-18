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

    def __init__(self, lm_type=1, predictor="model/shape_predictor_68_face_landmarks.dat", verbose=True):
        self.bbox_detector = dlib.get_frontal_face_detector()        
        self.landmark_predictor = dlib.shape_predictor(predictor)

        self.lm_2d_index = self.lm_2d_index_list[lm_type]
        self.landmarks_3d = self.landmarks_3d_list[lm_type]

        self.v = verbose
        
        self.concent = 100


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
            self.concent -= 1
            # 집중도 표시
            self.draw_concent(im)
            return im, None
        
        self.concent += 1

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