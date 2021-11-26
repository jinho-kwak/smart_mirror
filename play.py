import cv2
import numpy as np
import os
import sys
import glob
import imghdr
import random
import datetime
import time
from PIL import Image
import logging
import log_adapter



logger = logging.getLogger('console_file')
logger.info("=="*15+"[프로그램 시작]"+"=="*15)

def get_file_list(file_path):
    month_list = sorted(glob.glob(f'{file_path}/*'))
    show_file_list = []
    random.shuffle(month_list)

    for i in month_list:
        if os.path.isdir(i):
            show_file_list.extend(sorted(glob.glob(f'{i}/*')))
    return show_file_list

def get_last_access_time_list(file_list):
    tmp_list = []
    ttmp_list = []
    for idx, file in enumerate(file_list):
        atime = os.path.getctime(file)
        tmp_tuple = (time.ctime(atime),file)
        tmp_list.append(tmp_tuple)

    ### 아래 두개중 한개로 분기 태우기
    # tmp_list.sort(key=lambda x:x[0]) ## 시간 별로 소팅
    random.shuffle(tmp_list) ## 랜덤

    # list로 만들어 리턴 
    for i in tmp_list:
        ttmp_list.append(i[1])
    return ttmp_list   


def check_video_or_img(file):
    result = file.split('.')[1]
    return result

def myrange(start, end, step):
    r = start
    return_list = []
    while(r>end):
        return_list.append(round(r,2))
        r -= step
    return return_list


if __name__ == '__main__':
    argg = sys.argv[1]
    now_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = f'{now_dir}/{argg}'
    file_list = get_file_list(file_path) # 디렉토리안에 파일들 다 읽어서 file_list에 저장하기
    file_list = get_last_access_time_list(file_list)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # WINDOW_NORMAL 로 만들어야 전체화면 가능
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    logger.info(file_list)
    cnt = len(file_list)
    idx = 0
    while True:
        try:
            result_file = check_video_or_img(file_list[idx])
            result_file = result_file.upper()
            tmp_idx = 0
            # print(file_list[idx])

            # 이미지 분기 
            if result_file in ['JPG','PNG']:
                zero = np.zeros(shape=(1080,1920,3), dtype=np.uint8)
                img = cv2.imread(file_list[idx], cv2.IMREAD_COLOR)
                for i in myrange(1,0,0.02):
                    if img.shape[0]*i <= 1080 and img.shape[1]*i <= 1920:
                        tmp_idx = i
                        break

                img = cv2.resize(img, dsize=(0,0), fx=tmp_idx,fy=tmp_idx,interpolation=cv2.INTER_AREA)
                
                x = int((zero.shape[1] - img.shape[1]) / 2)
                y = int((zero.shape[0] - img.shape[0]) / 2)
                zero[y:y+img.shape[0], x:x+img.shape[1], :] = img
                cv2.imshow('image',zero)
                
                ch = cv2.waitKey(25)
                # 종료
                if ch == ord('q'):
                    break
            # 영상 분기 
            elif result_file in ['MOV','MP4']:
                MJPG_CODEC = 1196444237.0 # MJPG
                zero = np.zeros(shape=(1080,1920,3), dtype=np.uint8)
                cap = cv2.VideoCapture(file_list[idx])
                ret, img = cap.read()
                for i in myrange(1,0,0.02):
                    if img.shape[0]*i <= 1080 and img.shape[1]*i <= 1920:
                        tmp_idx = i
                        break
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width*tmp_idx)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height*tmp_idx)
                x = int((zero.shape[1] - img.shape[1]) / 2)
                y = int((zero.shape[0] - img.shape[0]) / 2)
                
                while True:
                    ret, img = cap.read()
                    if img is None:
                        break
                    
                    # x = int((zero.shape[1] - img.shape[1]) / 2)
                    # y = int((zero.shape[0] - img.shape[0]) / 2)
                    zero[y:y+img.shape[0], x:x+img.shape[1], :] = img
                    cv2.imshow('image',zero)
                    ch = cv2.waitKey(25)

                    # 종료
                    if ch == ord('q'):
                        break
                cap.release()

            else:
                print(f'확장자 {result_file} 분기 태울 것 ERROR')
            
            idx += 1
            if idx >= cnt:
                idx = 0
        except Exception as err:
            logger.warning(err)
        
    cv2.destroyAllWindows()
