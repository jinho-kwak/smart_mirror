import cv2
import numpy as np
import os
import sys
import glob
import time
import imutils 
def get_file_list(file_path):
    video_list = []

    tmp_list = sorted(glob.glob(f'{file_path}/*'))
    # img_list = sorted(glob.glob(f'{file_path}/*.jpg'))
    # img_list.extend(sorted(glob.glob(f'{file_path}/*.png')))
    # video_list = sorted(glob.glob(f'{file_path}/*.mp4'))
    return tmp_list

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
    file_list = get_file_list(file_path) # 디렉토리안에 파일들 다 읽어서 img_list에 저장하기
    print(file_list)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # WINDOW_NORMAL 로 만들어야 전체화면 가능
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.resizeWindow('image',1080,1920)
    

    cnt = len(file_list)
    idx = 0
    while True:
        result_file = check_video_or_img(file_list[idx])
        tmp_idx = 0
        print(file_list[idx])
        print(result_file)

        # 이미지 분기 
        if result_file in ['jpg','png']:
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
            if cv2.waitKey(3000) >= 0:
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
            print(width,height)
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
                ch = cv2.waitKey(1)

                # 종료
                if ch == ord('q'):
                    break
            cap.release()

        idx += 1
        if idx >= cnt:
            idx = 0
            break
    cv2.destroyAllWindows()
