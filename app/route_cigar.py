# -*- coding:utf-8 -*-
import enum
import os
import cv2
import redis
import time
import json
import ssl
import asyncio
import grpc
import logging
import requests
from sqlalchemy.engine.interfaces import ExceptionContext
# from tensorflow.python.keras.backend import set_value
from tensorflow.python.training import coordinator
# import torch
from app import models
import traceback
import copy
import logging.config
import numpy as np
import pandas as pd
# import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree, dump
import multiprocessing
import boto3
import ast

from keys import keys
from PIL import Image
from flask_cors import CORS
from .data_access import DAO
from datetime import datetime
from xml.dom import minidom
# from app.log_adapter import StyleAdapter
import app.log_adapter
# from multiprocessing import Process
from tensorflow.keras.models import load_model
from tensorflow import make_tensor_proto, make_ndarray
from flask import Flask, request, abort, render_template, redirect, Blueprint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from .config import Config, config_by_name
from .alarm import Alarm
from ec2_metadata import ec2_metadata

from .lc_inf import OrderList
# from .log_getter import LogGetter
from .log_designate import LogDesignate
from collections import Counter
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
import app.util as util
import threading
import base64
from .vision import ai_vision as ai
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#loop = asyncio.get_event_loop()

# Log 결정 Class
devkind = 'cigar'
log = LogDesignate(devkind)

log.info(f'{os.getpid()}|{devkind}_inference| ########## S T A R T ##########')

cigar_bp = Blueprint('main', __name__, url_prefix='/')

resize = (224, 224)
result_of_front_num = 3
s3 = boto3.client('s3')

# class RemoveDuplicatBox():
#     '''
#     바운딩 박스가 겹쳐졌을 경우, 겹쳐진 박스 중 한 개를 제거하는 로직
#     Created by JwMudfish
#     '''
#     def __init__(self, coor):
#         self.coor = coor
#         self.threshold = 0.1
#         self.limit_area = 2000

#     def calc_area(self, corr):
#         w = abs(corr[2] - corr[0])
#         h = abs(corr[3] - corr[1])
#         wh = int(w * h) // 100
#         return wh
    
#     def area_filter(self, coor):
#         del_list = list(filter(lambda x : self.calc_area(x) > self.limit_area, coor))
#         #log.info(f'대왕박스 발견 : {del_list}')
#         for i in del_list:
#             coor.remove(i)   
#         return coor

#     def iou(self, box1, box2):
#         box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#         box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
#         w = max(0, x2 - x1 + 1)
#         h = max(0, y2 - y1 + 1)
#         inter = w * h
#         iou = inter / (box1_area + box2_area - inter)
#         return iou

#     def compute_iou(self, coor):
#         box_list = copy.deepcopy(coor)
#         print('threshold :', self.threshold)
#         error_list = []
#         for _ in range(len(box_list)):
#             box_1 = box_list.pop()
#             for box_2 in box_list:
#                 result = self.iou(box_1, box_2)
#                 if result > self.threshold :
#                     error_list.append([box_1, box_2])
#                 # print(result)
#         return error_list

#     def listDupsUnique(self, lists):
#         return list(unique_everseen(duplicates(lists)))

#     def run(self):

#         dup_coor = copy.deepcopy(self.coor)
#         dup_coor = self.area_filter(coor = dup_coor)

#         error_list = self.compute_iou(self.coor)
#         tmp_list = sum(error_list, [])
#         print('첫 결과 :', len(dup_coor))
#         print('IOU 일정 이상 박스 : ',tmp_list)
        
#         del_list = self.listDupsUnique(tmp_list)
#         print('del_list : ', del_list)
        
#         for coor in del_list:
#             dup_coor.remove(coor)
#         print('최종 결과 :' , len(dup_coor))
        
#         tmp_error_list = self.compute_iou(dup_coor)
        
#         for i in tmp_error_list:
#             if i[0][1] < i[1][1]:
#                 i.remove(i[1])
#             else:
#                 i.remove(i[0])
        
#         try:
#             for i in range(len(tmp_error_list)):
#                 dup_coor.remove(tmp_error_list[i][0])
#         except:
#             pass

#         return dup_coor

# 냉장고 doorclosed Class
class Coordinate_refine:
    def __init__(self, companyId, storeId, deviceId, input_coor, re, width = 1920, height = 1080, margin = 330):
        self.companyId = companyId
        self.storeId = storeId
        self.deviceId = deviceId
        self.input_coor = input_coor
        self.width = width
        self.height = height
        self.margin = margin
        self.R_pnt = width//2 + margin
        self.L_pnt = width//2 - margin
        self.final_coor = []
        self.final_result = {}
        self.re = re
        self.save_xml_coor = []
        self.refine_coor()
        global s3
        self.s3 = s3
        self.infer_Bucket_name = 'smart-retail-inference'

    def refine_coor(self,):
        coor = [i for i in self.input_coor]
        coor = [list(map(lambda x : x[:4], i)) for i in coor]
        self.final_coor = [[list(map(int,i)) for i in k] for k in coor]

    def in_line(self, x, y, lr, line_coor):
        # x = x
        # y = y
        if lr == 'left':      # 오른편에 선 긋기 \
            x1, x2 = x, self.width
            y1, y2 = y, y+1
            # pt1 = self.R_pnt, 0
            # pt2 = self.width, self.height
            pt1 = line_coor['r'][0], line_coor['r'][1]
            pt2 = line_coor['r'][2], line_coor['r'][3]
            imgRect = (x,y,x2-x1, y2-y1)
            retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)
        elif lr == 'right':   # 왼편에 선긋기 /
            x1, x2 = x, self.width
            y1, y2 = y, y+1
            # pt1 = self.L_pnt, 0
            # pt2 = 0, self.height

            pt1 = line_coor['l'][0], line_coor['l'][1]
            pt2 = line_coor['l'][2], line_coor['l'][3]
            
            imgRect = (x,y,x2-x1, y2-y1)
            retval, rpt1, rpt2 = cv2.clipLine(imgRect, pt1, pt2)
        return retval
    def center_point(self, coor, xy = 'x'):
        if xy == 'x':
            #print('x : ', coor)
            min_coor, max_coor = coor[0], coor[2]
            center_point = (max_coor + min_coor) / 2
        elif xy == 'y':
            #print('y : ', coor)
            min_coor, max_coor = coor[1], coor[3]
            center_point = (max_coor + min_coor) / 2
        return int(center_point)

    def get_line(self, c_type, floor, index):
        cameras = Config.CAMERAS_LOCATION[c_type]
        line_coor = {}
        line_path = f'inference/lines/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{cameras[index]}'
        util.createFolder(line_path)
        try:
            line_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=f'{line_path}/line.xml')['Body']
            root = minidom.parse(line_data)

            line_l = root.getElementsByTagName('l')
            xleft = int(line_l[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_l[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_l[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_l[0].childNodes[7].childNodes[0].nodeValue)
            line_coor['l'] = [xleft,yleft,xright,yright]
            
            line_r = root.getElementsByTagName('r')
            xleft = int(line_r[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_r[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_r[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_r[0].childNodes[7].childNodes[0].nodeValue)
            line_coor['r'] = [xleft,yleft,xright,yright]
            try:
                line_m = root.getElementsByTagName('m')
                xleft = int(line_m[0].childNodes[1].childNodes[0].nodeValue)
                yleft = int(line_m[0].childNodes[3].childNodes[0].nodeValue)
                xright = int(line_m[0].childNodes[5].childNodes[0].nodeValue)
                yright = int(line_m[0].childNodes[7].childNodes[0].nodeValue)
                line_coor['m'] = [xleft,yleft,xright,yright]
            except:
                pass
        except Exception as err:
            log.info(f'{line_path}/line.xml 이 존재하지 않습니다. 기본 라인으로 설정합니다.\nERROR{err}')
            line_coor['l'] = [630,0,0,1080]
            line_coor['r'] = [1290,0,1920,1080]
            line_coor['m'] = [960,0,960,1080]
        return line_coor

    def split_normal_section(self, lcr, coor, c_type, floor, index):
        line_coor = self.get_line(c_type, floor, index)
        if c_type == 'NC':  # 일반 담배층일 경우
            s1 = int(line_coor['m'][0])
            section_1 = []
            section_2 = []
            for i in coor:
                x_coor = self.center_point(i, xy = 'x')
                y_coor = self.center_point(i, xy = 'y')
                if lcr == 'left':
                    if x_coor >= 0 and x_coor < s1:
                        section_1.append(i)
                    else:
                        tf = self.in_line(x = x_coor, y = y_coor, lr = 'left', line_coor = line_coor)
                        if tf == True:
                            section_2.append(i)
                elif lcr == 'right':
                    if x_coor > s1:
                        section_2.append(i)
                    else:
                        tf = self.in_line(x = x_coor, y = y_coor, lr = 'right', line_coor = line_coor)  # right
                        if tf == False:
                            section_1.append(i)
                elif lcr == 'center':
                    if x_coor <= s1:
                        tf = self.in_line(x = x_coor, y = y_coor, lr = 'right', line_coor = line_coor)
                        if tf == False:
                            section_1.append(i)
                    elif x_coor >= s1:
                        tf = self.in_line(x = x_coor, y = y_coor, lr = 'left', line_coor = line_coor)
                        if tf == True:
                            section_2.append(i)
            self.save_xml_coor.append(section_1+section_2)
            return section_1, section_2
        else:   # heats
            s1 = int(line_coor['l'][0])
            s2 = int(line_coor['r'][0])
            section_1 = []
            for i in coor:
                x_coor = self.center_point(i, xy = 'x')
                y_coor = self.center_point(i, xy = 'y')
                if x_coor > s1 and x_coor < s2:
                    section_1.append(i)
            self.save_xml_coor.append(section_1)
            return section_1

    def get_front_coor(self, section, num = result_of_front_num, total=False):
        rst = sorted(section, key=lambda x : self.center_point(x, xy='y'), reverse=True)
        if total == True:
            rst = rst
        else:
            rst = rst[:num]
        return rst

    def set_final_data(self, shelf_storage_dict):
        ## redis 에서 threshold, limit_area 값 가져오기
        try:
            removeBox_info = self.re.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_removebox_info')
            if removeBox_info:
                removeBox_info = eval(removeBox_info)
            else:
                removeBox_info = {"threshold" : 0.1, "limit_area" : 2000}
                log.info(f'removebox_info set {removeBox_info}')
                self.re.set(f'{self.companyId}_{self.storeId}_{self.deviceId}_removebox_info', str(removeBox_info))
        except Exception as err:
            log.warning(f"redis get threshold / limit_area ERROR {err}")
            removeBox_info = {"threshold" : 0.1, "limit_area" : 2000}
            self.re.set(f'{self.companyId}_{self.storeId}_{self.deviceId}_removebox_info', str(removeBox_info))

        idx = 0
        for key_f, value_type in shelf_storage_dict.items():
            final_list = []
            # value_type = 'NC' # test 용
            if value_type == 'NC':
                cameras = Config.CAMERAS_LOCATION[value_type]
                for index, coor_value in enumerate(self.final_coor[idx:idx + len(cameras)]):
                    rb = ai.RemoveDuplicatBox(coor = coor_value, threshold = removeBox_info['threshold'], limit_area = removeBox_info['limit_area'])
                    coor_value = rb.run()
                    if index == 0:
                        final_list.append(self.split_normal_section('left', coor_value, value_type, key_f, index))
                    elif index == 3:
                        final_list.append(self.split_normal_section('right', coor_value, value_type, key_f, index))
                    else:
                        final_list.append(self.split_normal_section('center', coor_value, value_type, key_f, index))
                idx += len(cameras)            
            else:### EC 일때 돌릴곳 
                cameras = Config.CAMERAS_LOCATION[value_type]
                for index, coor_value in enumerate(self.final_coor[idx:idx + len(cameras)]):
                    rb = ai.RemoveDuplicatBox(coor = coor_value, threshold = removeBox_info['threshold'], limit_area = removeBox_info['limit_area'])
                    coor_value = rb.run()
                    if index == 0:
                        final_list.append(self.split_normal_section('left', coor_value, value_type, key_f, index))
                    elif index == 5:
                        final_list.append(self.split_normal_section('right', coor_value, value_type, key_f, index))
                    else:
                        final_list.append(self.split_normal_section('center', coor_value, value_type, key_f, index))
                idx += len(cameras)
            self.final_result[key_f] = {'type': value_type}
            i = 1
            for cam_xy_coor in final_list:
                if isinstance(cam_xy_coor, list): ## hits 일때
                    self.final_result[key_f][f'columns_{i}'] = {'total_cnt': len(cam_xy_coor),
                                                            'total_coor' : cam_xy_coor, 
                                                            'front_cnt' : len(self.get_front_coor(cam_xy_coor)),
                                                            'front_coor' : self.get_front_coor(cam_xy_coor),
                                                        }
                    i += 1
                else:
                    for single_section in cam_xy_coor: ## 일반담배 일때 
                        self.final_result[key_f][f'columns_{i}'] = {'total_cnt': len(single_section),
                                                                'total_coor' : single_section, 
                                                                'front_cnt' : len(self.get_front_coor(single_section)),
                                                                'front_coor' : self.get_front_coor(single_section),
                                                            }
                        i += 1
        return self.final_result, self.save_xml_coor
    '''
    원본
    '''
    # def crop(self, img_list):
    #     '''
    #         img_list = [('naming', img )]
    #     '''
    #     img_list_each_column = []
    #     crop_list = []
    #     i = 0
    #     idx = 0
    #     for floor, value in self.final_result.items():
    #         for img in img_list[idx:idx + len(list(Config.CAMERAS_LOCATION[value['type']]))]:
    #             image = np.frombuffer(img[1], dtype=np.uint8) 
    #             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #             img_list_each_column.append(image)
    #             if value['type'] == 'NC':
    #                 img_list_each_column.append(image)
    #         idx += len(list(Config.CAMERAS_LOCATION[value['type']]))

    #         for index, columns_i in enumerate(list(value.keys())[1:]):
    #             images = list(map(lambda b : img_list_each_column[i][b[1]:b[3], b[0]:b[2]], value[columns_i]['front_coor']))
    #             images = list(map(lambda d : cv2.resize(d, resize), images))
    #             crop_list.extend(images)
    #             i += 1
    #     return crop_list

    '''
    1번 멀티프로세싱 crop img_list
    '''
    def crop_worker(self, crop_return_dict, crop_return_dict_origin, floor, value, img_list):
        start_time = time.time()
        img_list_each_column = []
        crop_img_list = []
        crop_img_list_origin = []
        for img in img_list:
            image = np.frombuffer(img[1], dtype=np.uint8) 
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_list_each_column.append(image)
            if value['type'] == 'NC':
                img_list_each_column.append(image)
        for index, columns_i in enumerate(list(value.keys())[1:]):
            origin_images = list(map(lambda b : img_list_each_column[index][b[1]:b[3], b[0]:b[2]], value[columns_i]['front_coor']))
            images = list(map(lambda d : cv2.resize(d, resize), origin_images))
            crop_img_list_origin.extend(origin_images)
            crop_img_list.extend(images)
        crop_return_dict_origin[floor] = crop_img_list_origin
        crop_return_dict[floor] = crop_img_list
    def crop(self, img_list):
        '''
            img_list = [('naming', img )]
        '''
        img_list_each_column = []
        crop_list = []
        crop_list_origin = []
        crop_manager = multiprocessing.Manager()
        crop_return_dict = crop_manager.dict()
        crop_return_dict_origin = crop_manager.dict()
        crop_jobs = []
        idx = 0
        for floor, value in self.final_result.items():
            p1 = multiprocessing.Process(target=self.crop_worker, args=(crop_return_dict, crop_return_dict_origin, floor, value, img_list[idx:idx + len(list(Config.CAMERAS_LOCATION[value['type']]))]))
            crop_jobs.append(p1)
            p1.start()
            idx += len(list(Config.CAMERAS_LOCATION[value['type']]))

        for crop_proc in crop_jobs:
            crop_proc.join()
        
        for img_key, img_value in sorted(crop_return_dict.items()):
            crop_list.extend(img_value)
        for img_key, img_value in sorted(crop_return_dict_origin.items()):
            crop_list_origin.extend(img_value)

        return crop_list, crop_list_origin


class CigarDoorClosed:
    def __init__(self, companyId, storeId, deviceId, work_user, trDate, trade_date, trade_time, trNo, trResponseDate):
        '''
            파라미터
                companyId: 고객사 코드
                storeId: 점포 코드
                deviceId: 냉장고 코드
                dao: DB 접근 하기 위한 파이썬 모듈
                work_user:
                    customer: 고객
                    manager: 냉장고 관리자
                    interminds : 인터마인즈 직원

        '''
        self.re = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
                            db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
                            password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD, charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
                            decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)

        self.re_img = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
                db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
                password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD)
        self.obj_url = f'http://{config_by_name[Config.BOILERPLATE_ENV].EC2_OBJECT_DETECTION_IP}:{config_by_name[Config.BOILERPLATE_ENV].EC2_OBJECT_DETECTION_PORT}/predict/ciga'
        global s3
        self.s3 = s3 
        self.dao = DAO()
        self.log_dao = DAO()
        self.companyId = companyId
        self.storeId = storeId
        self.deviceId = deviceId
        self.work_user = work_user
        self.trade_date = trade_date
        self.trNo = trNo
        self.trade_time = trade_time
        self.trDate = trDate
        self.trResponseDate = trResponseDate
        self.infer_save_img_time = None
        self.message = Alarm(companyId, storeId, deviceId, work_user)
        self.pre_trDate = self.re.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_trDate')
        self.count_vision_log = ""
        self.infer_Bucket_name = 'smart-retail-inference'
        self.save_img_Bucket_name = 'smart-retail-server-log'
        self.save_log_to_s3 = config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3
        self.conf_email_alarm = config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM
        self.cigar_cells_list = []
        self.cigar_cells_stock = []
        self.cnt_dict = {}
        self.shelf_storage_dict = {}
        self.main_df = []
        self.device_shelf_list = self.dao.get_device_shelf(self.companyId, self.storeId, self.deviceId)

    def __del__(self):
        self.dao.session.close()

    def send_alimtalk_msg(self, error_type, msg):
        if config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM == True and msg != "":
            if self.work_user == 'manager':
                header = "(관리자 모드)"
                send_mode = "default"
            elif self.work_user == 'interminds':
                header = "(테스트 모드)"
                send_mode = "interminds"
            else:
                header = "(사용자 모드)"
                send_mode = "default"
            self.message.send_alimtalk(error_type, self.trDate, msg, send_mode, header)


    def check_mininum_stocks(self, order_list):
        '''
            셀의 재고가 알람 개수 이하면 카톡 알림
            (재고가 알람 개수 초과에서 이하로 변경된 경우, 재고 모드가 abs인 경우, 무게 오류 발생하지 않은 경우만 알림)
            섞인 상품은 제외
        '''
        # get minium stock
        # ("층", "칸", "바코드", "상품명", "현재 재고", "알림 개수")
        stock_list = self.dao.get_cigar_stocks_by_csd_id(self.companyId, self.storeId, self.deviceId)
        # if not exist minium stock -> return
        if len(stock_list) == 0 or len(order_list) == 0:
            return

        # need reverse floor
        device_pkey = self.dao.get_device_pkey(self.companyId, self.storeId, self.deviceId)
        floor = self.dao.get_max_floor(device_pkey)

        alarm_msg = ""
        for line in stock_list:
            line = list(line)
            alarm_check = [ol for ol in order_list if int(ol['rowNo'])==line[0] and int(ol['colNo'])==line[1] and int(ol['goodsCnt'])+line[4] > line[5]]
            if alarm_check :
                line[0] = floor-line[0]+1
                line[1] += 1
                alarm_msg += "- {0}층 {1}칸 {3}({2}) {4}개 ({5}개 이하 알림)\n\n".format(*line)
        if config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM == True and alarm_msg != "":
            if self.work_user == 'manager':
                header = "(관리자 모드)"
                send_mode = "default"
            elif self.work_user == 'interminds':
                header = "(테스트 모드)"
                send_mode = "interminds"
            else:
                header = "(사용자 모드)"
                send_mode = "default"
            # self.message.send_alimtalk("stock_alarm", self.trDate, alarm_msg, send_mode, header)
            self.message.send_slack(self.trDate,alarm_msg,devkind)

    def save_img_s3_or_local(self, save_path, image):
        if self.save_log_to_s3 == True:
            self.s3.put_object(Body=cv2.imencode('.jpg', image)[1].tostring(), Bucket=self.save_img_Bucket_name, Key=save_path)
        else:
            cv2.imwrite(f'{save_path}', image)


    '''
    1번 멀티프로세싱 층별로 프로세스 나누기
    '''
    def get_img_list_worker(self, return_dict, floor, cameras):
        start_time = time.time()
        img_list = []
        try:
            for camera in cameras:
                saved_full_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}/{self.trDate.split("_")[0]}'
                img = self.re_img.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_f{floor}_cam{camera}')
                encoded_img = np.frombuffer(img, dtype=np.uint8) 
                image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.shelf_storage_dict[floor] == 'EC':
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 
                t1 = threading.Thread(target=self.save_img_s3_or_local, args=(f'{saved_full_path}/{self.trDate}.jpg', image,))
                t1.daemon = True 
                t1.start()
                # self.save_img_s3_or_local(f'{saved_full_path}/{self.trDate}.jpg', image)
                # _, frame = cv2.imencode('.jpg', image)
                # image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ##인퍼런스결과확인후에 결정 할 것 
                img_2_byte = cv2.imencode('.png',image)[1].tobytes()
                img_list.append(('image',img_2_byte))
            return_dict[floor] = img_list
        except Exception as err:
            log.error(traceback.format_exc())

    def get_img_list(self, ):
        start_time = time.time()
        final_img_list = []
        self.shelf_storage_dict = {}
        get_img_list_manager = multiprocessing.Manager()
        return_dict = get_img_list_manager.dict()
        get_img_list_jobs = []
        for index, device_shelf in self.device_shelf_list.iterrows():
            floor = device_shelf['shelf_floor']
            shelf_storage_type = device_shelf['shelf_storage_type'] # EC / NC
            shelf_storage_type = shelf_storage_type.split()[0]
            self.shelf_storage_dict[floor] = shelf_storage_type
            # saved_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'
            cameras = Config.CAMERAS_LOCATION[shelf_storage_type]
            p = multiprocessing.Process(target=self.get_img_list_worker,args=(return_dict, floor, cameras))
            get_img_list_jobs.append(p)
            p.start()
        for get_img_list_proc in get_img_list_jobs:
            get_img_list_proc.join()
        for img_key, img_value in sorted(return_dict.items()):
            final_img_list.extend(img_value)
        return final_img_list, self.shelf_storage_dict

    # '''
    # 원본 함수 
    # '''
    # def get_img_list(self, ):
    #     start_time = time.time()
    #     img_list = []
    #     self.shelf_storage_dict = {}
    #     for index, device_shelf in self.device_shelf_list.iterrows():
    #         floor = device_shelf['shelf_floor']
    #         shelf_storage_type = device_shelf['shelf_storage_type'] # EC / NC
    #         self.shelf_storage_dict[floor] = shelf_storage_type
    #         saved_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'
    #         # util.createFolder(saved_path)
    #         ## config에서 camera_location 뽑기 
    #         cameras = Config.CAMERAS_LOCATION[shelf_storage_type]
    #         for camera in cameras:
    #             try:
    #                 img = self.re_img.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_f{floor}_cam{camera}')
    #                 save_full_path = f'{saved_path}/{camera}/{self.trDate.split("_")[0]}'  # 이미지 저장소 파일(날짜별)
    #                 # util.createFolder(save_full_path)
    #                 encoded_img = np.frombuffer(img, dtype=np.uint8) 
    #                 image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    #                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #                 if shelf_storage_type == 'EC':
    #                     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 
    #                 self.save_img_s3_or_local(f'{save_full_path}/{self.trDate}.jpg', image)
    #                 _, frame = cv2.imencode('.jpg', image)
    #                 image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    #                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #                 img_2_byte = cv2.imencode('.png',image)[1].tobytes()
    #                 img_list.append(('image',img_2_byte))
    #             except Exception as err:
    #                 log.error(f'cameras images redis get Error({str(err)})')
    #     return img_list, self.shelf_storage_dict

    def product_count(self, final_result):
        orderlist = {}
        total_order = []
        out_order_list = []
        in_order_list = []
        total_log = {}
        stock_count = {}
        self.cnt_dict = {}
        self.cigar_cells_list = []
        floors = list(final_result.keys())
        goods_log = [[] for f in floors]
        
        # get cell_alert_dict
        cell_alert_dict = self.re.get('{}_{}_{}_cell_alert'.format(self.companyId, self.storeId, self.deviceId))
        if cell_alert_dict:
            cell_alert_dict = ast.literal_eval(cell_alert_dict)
        else:
            cell_alert_dict = {}
            for floor in floors:
                cell_alert_dict[str(floor)] = {}

        for floor, value in final_result.items():
            for index, columns_i in enumerate(list(value.keys())[1:]):
                master_d_pkey = self.dao.get_cell_and_design_keys(self.companyId, self.storeId, self.deviceId, floor, index)
                # case1. object detection 박스 갯수가 최대 max_count 보다 많으면 cell_error
                if value[columns_i]['total_cnt'] > master_d_pkey['stock_count_max']:
                    cell_alert_dict[str(floor)][str(index)] = 1
                else:
                    cell_alert_dict[str(floor)][str(index)] = 0
                self.cnt_dict[master_d_pkey['cell_pkey']] = {}

                # label to cell 의 design_master_pkey 로 변환 
                for infer_index, infer_label in enumerate(value[columns_i]['front_result']):
                    value[columns_i]['front_result'][infer_index] = self.dao.get_design_pkey(design_infer_label = infer_label)
                    
                if value[columns_i]['total_cnt'] >= value[columns_i]['front_cnt']:
                    self.cnt_dict[master_d_pkey['cell_pkey']][master_d_pkey['design_pkey_master']] = value[columns_i]['total_cnt'] - value[columns_i]['front_cnt'] 



                for j in final_result[floor][columns_i]['front_result']:
                    self.cnt_dict[master_d_pkey['cell_pkey']][j] = self.cnt_dict[master_d_pkey['cell_pkey']].get(j,0) + 1
                
                # Cell_pkey Setting. (cigar_cells)
                cigar_cells_dict = {}
                cigar_cells_dict.setdefault('cell_pkey', None)
                cigar_cells_dict.setdefault('d_p_m', None)
                cigar_cells_dict.setdefault('d_p_f', None)
                cigar_cells_dict.setdefault('d_p_s', None)
                cigar_cells_dict.setdefault('d_p_t', None)
                cigar_cells_dict.setdefault('total_cnt', None)

                cigar_cells_dict['cell_pkey'] = master_d_pkey['cell_pkey']
                cigar_cells_dict['d_p_m'] = master_d_pkey['design_pkey_master']
                try:
                    cigar_cells_dict['d_p_f'] = value[columns_i]['front_result'][0]
                    cigar_cells_dict['d_p_s'] = value[columns_i]['front_result'][1]
                    cigar_cells_dict['d_p_t'] = value[columns_i]['front_result'][2]
                except Exception as err:
                    pass
                cigar_cells_dict['total_cnt'] = value[columns_i]['total_cnt']
                self.cigar_cells_list.append(cigar_cells_dict)
        '''
        bf_stock - af_stock 해서 orderlist 만드는 곳 
        orderlist = {cell_pkey1 : {design_pkey1: cnt, design_pkey2: cnt, design_pkey3: cnt}, 
                    cell_pkey2 : {design_pkey1: cnt, design_pkey2: cnt, design_pkey3: cnt}
                    ...}
        '''
        for cell_pkey, af_stock in self.cnt_dict.items():
            orderlist[cell_pkey] = {}
            bf_stock = self.dao.get_cigar_stocks(cell_pkey)
            for i,j in bf_stock.items():
                orderlist[cell_pkey][i] = j
            for i,j in af_stock.items():
                orderlist[cell_pkey][i] = bf_stock.get(i,0) - j
        '''
        cnt_dict에 pkey를 goods_id로 바꿔서 복사하기 trade_log 용
        '''
        for cell_pkey, stock_cnt in self.cnt_dict.items():
            stock_count[cell_pkey] = {}
            for pkey, cnt in stock_cnt.items():
                goods_id = self.dao.get_designs_by_design_pkey(pkey)
                stock_count[cell_pkey][goods_id.goods_id] = cnt
        
        '''
        orderlist로 포맷맞추기 -> total_order
        total_order = [{'goodId': -- , 'goodName': -- , 'RowNo' : -- , 'ColNo': --, 'goodsCnt': --}, {...}, {...}, ...]
        '''
        for cell_pkey, r in orderlist.items():
            trade_goods_name_value = {}  # for log
            for pkey, cnt in r.items():
                if cnt != 0:
                    design_pkey = self.dao.get_designs_by_design_pkey(str(pkey))
                    floor_cell_data = self.dao.get_cell_column_shelf_floor(cell_pkey)
                    goods_price = self.dao.get_sale_price(self.storeId, str(pkey))
                    order_dict = {'goodsId':design_pkey.goods_id,
                                        'goodsName':self.dao.get_goods_name(design_pkey.goods_id),
                                        'goodsCnt':str(cnt),
                                        'goodsPrice':None if goods_price is None else str(goods_price),
                                        'rowNo':floor_cell_data['shelf_floor'],
                                        'colNo':floor_cell_data['cell_column'],
                                }
                    total_order.append(order_dict)
                    if cnt > 0:
                        out_order_list.append(order_dict)
                    else:
                        in_order_list.append(order_dict)
                    
                    # db 로그테이블 insert
                    self.log_dao.insert_cigar_trade_log(
                        cigar_trade_log_no = self.trNo,
                        cigar_trade_log_date = self.trade_date,
                        cigar_trade_log_time = self.trade_time,
                        company_id = self.companyId,
                        store_id = self.storeId,
                        device_id = self.deviceId,
                        shelf_floor = floor_cell_data['shelf_floor'],
                        cell_column = floor_cell_data['cell_column'],
                        goods_id = design_pkey.goods_id,
                        goods_name = self.dao.get_goods_name(design_pkey.goods_id),
                        goods_label = design_pkey.design_infer_label,
                        goods_count = str(cnt),
                        stock_left = str(stock_count[cell_pkey]),
                        duration = None,
                        work_user = self.work_user,
                        work_type = "trade",
                        status_code = None,
                        sale_price = None,
                        total_sale_price = None,
                    )
                    self.log_dao.session.commit()
                    trade_goods_name_value[self.dao.get_goods_name(design_pkey.goods_id)] = trade_goods_name_value.get(self.dao.get_goods_name(design_pkey.goods_id), 0) + cnt
            goods_log[self.dao.get_cell_column_shelf_floor(cell_pkey)['shelf_floor']].append(trade_goods_name_value)
        
        for row in goods_log:
            for col in row:
                del_list = []
                for key ,value in col.items():
                    # delete zero log
                    if value == 0:
                        del_list.append(key)
                    # calculate total log
                    if key in total_log.keys():
                        total_log[key] += value
                    else:
                        total_log[key] = value
                for key in del_list:
                    del col[key]
                        
        total_log_d = total_log.copy()
        for key, value in total_log.items():
            if value == 0:
                del total_log_d[key]
        
        # total_log 상품이 -1 이 있으면 해당 상품 이름과, 갯수를 알림보냄
        msg = ""
        for key, value in total_log_d.items():
            if value < 0:
                msg += f"상품명 : {key} 개수 : {value} 재고 확인요망\n"
        if self.work_user == 'customer':
            self.message.send_slack(self.trDate , msg, devkind)
            # self.send_alimtalk_msg("stock_alarm", msg)



        log_str = """
------------ goods -------------
""" + "\n".join(list(map(lambda row: " | ".join([f"#{i} "+f"{cell if cell else ''}".center(8, " ") for (i, cell) in enumerate(row)]), goods_log))) + f"""

------------ total -------------
{total_log_d if total_log_d else 'nothing in & out'}

---------- order list ----------
"""
        if total_order:
            if out_order_list:
                log_str += "- OUT -" + "\n"
                for out_data in out_order_list:
                    log_str += str(out_data) + "\n"
            if in_order_list:
                log_str += "- IN -" + "\n"
                for in_data in in_order_list:
                    log_str += str(in_data) + "\n"
        else:
            log_str += 'no orderlist' + "\n" 

        util.LogGetter.log += log_str

        # 기본 로그들 S3에 저장
        all_floor_save_img_path = {}
        for floor, storage_type in self.shelf_storage_dict.items():
            floor_save_img_path = []
            cameras = Config.CAMERAS_LOCATION[storage_type]
            for camera in cameras:
                floor_save_img_path.append(f'logs/saved_box_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}/{self.trDate.split("_")[0]}/{self.trDate}.jpg')
            all_floor_save_img_path[str(floor)] = floor_save_img_path
        
        admin_json_text = {
            'trNo' : self.trNo,
            'trade_date' : str(self.trade_date),
            'trade_time' : str(self.trade_time),
            'companyId' : self.companyId,
            'storeId' : self.storeId,
            'deviceId' : self.deviceId,
            'floors' : str(len(self.shelf_storage_dict.keys())),
            'request.path' : request.path,
            'request.url' : request.url,
            'request.remote_addr' : request.remote_addr,
            'request.method' : request.method,
            'request.data' : json.loads(request.data),
            'payment_error' : "False",
            'all_floor_save_img_path' : all_floor_save_img_path,
            'count_log' : util.LogGetter.log,
        }

        save_admin_json_path = f'logs/log/payment/{self.companyId}/{self.storeId}/{self.deviceId}/{self.trDate.split("_")[0]}'
        if self.save_log_to_s3 == True:
            self.s3.put_object(Body=bytes(json.dumps(admin_json_text, indent=4, ensure_ascii=False).encode('UTF-8')),
                                    Bucket=f'{self.save_img_Bucket_name}', Key=f'{save_admin_json_path}/{self.trDate.split("_")[-1]}.txt')
        else:
            if not os.path.exists(save_admin_json_path):
                os.makedirs(save_admin_json_path)
            with open(f'{save_admin_json_path}/{self.trDate.split("_")[-1]}.txt', 'wt') as json_file:
                json.dump(admin_json_text, json_file, indent=4)

        self.re.set('{}_{}_{}_cell_alert'.format(self.companyId, self.storeId, self.deviceId), str(cell_alert_dict))

        return total_order

    def infer(self, crop_img_list):
        try:
            model_txt = self.device_shelf_list.shelf_model.values[0]
            
            ## jinho start
            model_label_path = f'inference/model_and_label/{model_txt}/main_labels.txt'
            if self.save_log_to_s3 == True:
                model_label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=model_label_path)['Body']
                self.main_df = pd.read_csv(model_label_data, sep=' ', index_col=False, header=None)
            else:
                self.main_df = pd.read_csv(model_label_path, sep=' ', index_col=False, header=None)
            ## --- jinho end

            try:
                api_info = self.re.get(f'cigar_api_info')
                if api_info:
                    api_info = eval(api_info)
                else:
                    log.warning(f'[cigar_api_info is None] Infer Redis Get Error({str(err)}) | 강제 api_info 적용 ({api_info})')
                    api_info['INFERENCE_URL'] = 'https://125.132.250.227/v0/vision_api'
                    api_info['INFERENCE_KEY'] = 'pTsHkPdCyeFDolBRywQTcXurdbxBL7K2kstMoTHs'
                    api_info['INFERENCE_USERID'] = 'jspark'
            except Exception as err:
                l
                api_info['INFERENCE_URL'] = 'https://125.132.250.227/v0/vision_api'
                api_info['INFERENCE_KEY'] = 'pTsHkPdCyeFDolBRywQTcXurdbxBL7K2kstMoTHs'
                api_info['INFERENCE_USERID'] = 'jspark'
                log.warning(f'[cigar_api_info] Infer Redis Get Error({str(err)}) | 강제 api_info 적용 ({api_info})')

            new_pay = [base64.b64encode(cv2.imencode('.jpg', c)[1].tobytes()).decode('utf-8') for c in crop_img_list]
            predict_product_name = requests.post(api_info['INFERENCE_URL'], data=json.dumps(new_pay),
                                                headers={"X-API-KEY": api_info['INFERENCE_KEY'],
                                                         "Userid": api_info['INFERENCE_USERID']}, verify=False)
            if '404' in str(predict_product_name):
                raise Exception(f'infer Error (Response [404])')
        except Exception as err:
            log.error(f'infer Error ({str(err)}) / predict_product_name : {predict_product_name}')
            log.error(traceback.format_exc())
            raise Exception(f'infer Error({err})')

        # predict_img = np.array(crop_img_list)
        # log.info(f'predict_img : {predict_img[0].shape}')
        # channel = grpc.insecure_channel(f'{config_by_name[Config.BOILERPLATE_ENV].EC2_INFERENCE_IP}:{config_by_name[Config.BOILERPLATE_ENV].EC2_INFERENCE_PORT}')
        # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        # request = predict_pb2.PredictRequest()
        # request.model_spec.name = f'{model_txt}_main' 
        # request.inputs['input_1'].CopyFrom(
        #     make_tensor_proto(predict_img, shape=predict_img.shape, dtype=float))
        # result = stub.Predict(request, 60.0)
        # outputs = result.outputs
        # detection_classes = outputs["dense"]
        # detection_classes = make_ndarray(detection_classes)
        # score = np.argmax(detection_classes, axis=1)
        # model_label_path = f'inference/model_and_label/{model_txt}/main_labels.txt'
        # ####
        # if self.save_log_to_s3 == True:
        #     model_label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=model_label_path)['Body']
        #     self.main_df = pd.read_csv(model_label_data, sep=' ', index_col=False, header=None)
        # else:
        #     self.main_df = pd.read_csv(model_label_path, sep=' ', index_col=False, header=None)
        # main_class_names = sorted(self.main_df[0].tolist())
        # predict_product_name = list(map(lambda x: main_class_names[x], score))

        return predict_product_name.json()


    # inference bottom error check
    def infer_bottom_check(self, crop_img_list_origin, infer_result, first_idx, end_idx, floor, cell):
        for label_index, infer_label in enumerate(infer_result[first_idx:first_idx + end_idx]):
            log_str = f'[({floor})floor/({cell})cell/({label_index})order]: '
            if infer_label == 'bottom':
                try:
                    decoded = pyzbar.decode(crop_img_list_origin[first_idx + label_index])
                    if decoded:
                        rst = decoded[0].data.decode("utf-8")
                    else:
                        raise Exception(f'barcode(None)에 해당하는 (design_label)이 존재하지 않습니다. ')
                    
                    design_label = self.dao.get_design_label(rst, self.main_df[0].tolist())
                    # 정상 case
                    if design_label:
                        log_str += f'goods_id({rst})에 해당하는 (\"{design_label}\")로 변환 하였습니다.'
                        infer_result[first_idx + label_index] = design_label
                    else:
                        raise Exception(f'goods_id({rst})에 해당하는 (design_label)이 존재하지 않습니다. ')
                except Exception as err:
                    log_str += str(err)
                    cell_design_keys_dict = self.dao.get_cell_and_design_keys(self.companyId, self.storeId, self.deviceId, floor, cell)
                    db_master_pkey = self.dao.get_goods_by_design_pkey(cell_design_keys_dict['design_pkey_master'])
                    infer_result[first_idx + label_index] = db_master_pkey.design_infer_label
                    log_str += f'[강제 pkey 변환 (\"{db_master_pkey.design_infer_label}\")]'
            else:
                continue
            log.warning(log_str)
        return infer_result
    
    def infer_vision_check(self,):
        pass

    
@cigar_bp.route('/cigar_model', methods=['POST'])
def tf_model():
    tmp_start = time.time()
    try:
        companyId = request.json['companyId']
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        try:
            trDate = request.json['trDate']
        except:
            trDate = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        trade_date = trDate.split('_')[0]
        trade_time = trDate.split('_')[1]

        try:
            trResponseDate = request.json['trResponseDate']
        except:
            trResponseDate = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            work_user = request.json['work_user']
        except:
            work_user = 'customer'
        trNo = request.json['trNo']
        #make return format
        result = {}
        result['abort'] = {'code': 200, 'msg': ''}
        result['trDate'] = trResponseDate
        result['trNo'] = trNo
        result['orderList'] = []
        log_str = '\n- Inference info -\n'
        log_str += f'companyId:{companyId}|storeId:{storeId}|deviceId:{deviceId}\n'
        log_str += f'trNo:{trNo}|trDate:{trDate}|work_user:{work_user}\n'
        # Db class
        dao = DAO()

        ## door close class
        start_time = time.time()
        door_closed = CigarDoorClosed(companyId, storeId, deviceId, work_user,
                                       trDate, trade_date, trade_time, trNo, trResponseDate)
        # redis에서 전체 층의 이미지를 리스트로 가져옴
        # img_list = [img1,img2,img3 , ....]   shelf_storage_dict = { f0 : 'shelf_storage_type', f1 : 'shelf_storage_type' , ... } 
        try: 
            img_list, shelf_storage_dict = door_closed.get_img_list()
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f"[get_img_list] : {str(err)}")
        log.info(f'---{time.time() - start_time}- door_closed.get_img_list')
        # obj 하는 request
        try:
            start_time = time.time()
            try:
                api_info = door_closed.re.get(f'cigar_api_info')
                api_info = eval(api_info)
            except Exception as err:
                api_info['OBJECT_DETECTION_URL'] = 'https://125.132.250.227/predict/ciga'
                api_info['OBJECT_DETECTION_KEY'] = 'eUAg9RfZ5AtoUejUr6uquTtFCeZ7vBpf9sxoNgzM'
                api_info['OBJECT_DETECTION_USERID'] = 'DevOps'
                log.warning(f'[cigar_api_info] object Redis Get Error({str(err)}) | 강제 api_info 적용 ({api_info})')
            try:
                response = requests.post(api_info['OBJECT_DETECTION_URL'], files=img_list, 
                                        headers={"X-API-KEY": api_info['OBJECT_DETECTION_KEY'], 
                                                "Userid": api_info['OBJECT_DETECTION_USERID']}, verify=False)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                log.error(f'obj request Error ({str(e)})')
                response = requests.post(api_info['OBJECT_DETECTION_URL'], files=img_list, 
                                        headers={"X-API-KEY": api_info['OBJECT_DETECTION_KEY'], 
                                                "Userid": api_info['OBJECT_DETECTION_USERID']}, verify=False)

            log.info(f'---{time.time() - start_time}- requests of object detection')

        except Exception as err:
            traceback.print_exc()
            raise Exception(f'[request.post(door_closed.obj_url)] : {str(err)}')

        # 좌표값 정제 
        try:
            start_time = time.time()
            refine = Coordinate_refine(companyId, storeId, deviceId, response.json()['res'], door_closed.re)  
            final_result, xml_coor = refine.set_final_data(shelf_storage_dict)
            
            # 보완로직 (박스 중복 제거 로직)
            # 섹션 마다 옵션처리? 모드 설정 (중복제거 로직 여부)

            log.info(f'---{time.time() - start_time} - refine coordinate')
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[refine.set_final_data] : {str(err)}')
        
        ##이미지 크롭 crop_img_list = [img1,img2,img3, ...]
        try:
            start_time = time.time()
            crop_img_list, crop_img_list_origin = refine.crop(img_list)  ## 이미지 크롭하는 곳
            log.info(f'---{time.time() - start_time} - crop_img_list')
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[refine.crop] : {str(err)}')

        # 좌표 xml로 저장하기 / img에 xml 붙여서 저장하기 
        try:
            start_idx = 0
            start_time = time.time()
            save_xml_idx = 0
            for floor, cigar_type in shelf_storage_dict.items():
                saved_xml_path = f'inference/boxes/{companyId}/{storeId}/{deviceId}/{floor}'
                saved_img_path = f'logs/saved_box_img/{companyId}/{storeId}/{deviceId}/{floor}'
                util.createFolder(saved_xml_path)
                util.createFolder(saved_img_path)
                for index, camera in enumerate(Config.CAMERAS_LOCATION[cigar_type]):
                    full_saved_xml_path = f'{saved_xml_path}/{camera}/{trDate.split("_")[0]}'
                    full_saved_img_path = f'{saved_img_path}/{camera}/{trDate.split("_")[0]}'
                    # t1 = threading.Thread(target=util.save_img_xml_s3_or_local, args=(f'{full_saved_img_path}/{trDate}.jpg', xml_coor[index+save_xml_idx], s3, config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3, 'smart-retail-server-log', shelf_storage_dict[floor], door_closed.re_img))
                    t1 = threading.Thread(target=util.save_img_xml_s3_or_local, args=(f'{full_saved_img_path}/{trDate}.jpg', xml_coor[index+save_xml_idx], s3, config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3, 'smart-retail-server-log', shelf_storage_dict[floor], img_list[start_idx]))
                    t1.daemon = True 
                    t1.start()
                    t2 = threading.Thread(target=util.save_xml, args=(full_saved_xml_path, trDate, xml_coor[index+save_xml_idx], s3, config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3, 'smart-retail-inference'))
                    t2.daemon = True 
                    t2.start()
                    start_idx += 1 
                save_xml_idx += len(Config.CAMERAS_LOCATION[cigar_type])
            log.info(f'---{time.time() - start_time} - save xml')
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'save_xml :{str(err)}')

        ###인퍼런스 돌려돌려 돌림판
        try:
            start_time = time.time()
            ### 크롭할 이미지가 없으면 inference안함
            if crop_img_list:
                infer_result = door_closed.infer(crop_img_list)
            else:
                infer_result = []
            log.info(f'---{time.time() - start_time} - infer')
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[door_closed.infer] : {str(err)}')

        # #bottom 일때 goods_id 에 매칭시켜서 goods_name으로 변경
        # for index, infer_label in enumerate(infer_result):
        #     try:
        #         if infer_label == 'test_bottom':
        #             decoded = pyzbar.decode(crop_img_list[index])
        #             rst = decoded[0].data.decode("utf-8")
        #             infer_result[index] = dao.get_goods_name(rst)
        #     except Exception as err:
        #         log.error(f'ERROR - {err} index {index} {infer_label}')

        #         raise Exception(f'[bottom 일때 goods_id에 매칭해서 goods_name으로 변경] : {str(err)}')


        # ##결과값 label로 나오는거 pkey값으로 바꾸기
        # try:
        #     infer_label = []
        #     for index, label in enumerate(infer_result):
        #         infer_label.append(dao.get_design_pkey(design_infer_label = label))
        # except Exception as err:
        #     log.error(f'label to pkey error - index : {index}, label : {label}')
        #     raise Exception(f'label to pkey error : {str(err)}')
        
        '''
        inference_mode 가 lc 이면 강제로 infer_label변경해서 대입한다.
        infer_result 가 bottom이면 바코드를 goods_id에서 찾아서 inference_label로 변환하여 front_result에 저장한다.
        만약 카메라의 포커스문제로인하여 바코드 변환이 불가능하거나 DB에서 해당 바코드를 찾지못하면 
        해당cell의 design_pkey_master로 변환하여 inference_label로 바꿔 front_result에 저장한다.
        
        이후 product_count()에서 inference_label인 front_result를 design_pkey로 변환하여 로직을 실행한다.
        '''

        
        start_time = time.time()
        first_idx = 0
        log_label = {}
        device_pkey = dao.get_device_pkey(companyId, storeId, deviceId)
        max_floor = dao.get_max_floor(device_pkey)
        # 정상 case, error case(2) 처리
        pog_list_by_design_infer_label = dao.get_ciga_pog_goods(companyId, storeId, deviceId)['design_infer_label'].tolist()
        for floor, value in final_result.items():
            log_label[floor] = {}
            floors = max_floor-floor+1
            cells_master_by_floor = dao.get_designs_pkey_master_cigar(companyId, storeId, deviceId, floor)
            compare_list_master = []
            for master_value in list(cells_master_by_floor.values()):
                compare_list_master.append(master_value)
            for cell, columns_i in enumerate(list(value.keys())[1:]):
                end_idx = value[columns_i]['front_cnt']
                # case0. LC mode
                if compare_list_master[cell]['inference_mode'] == 'lc':
                    for index, contain in enumerate(infer_result[first_idx:first_idx+end_idx]):
                        infer_result[first_idx+index] = compare_list_master[cell]['design_infer_label']
                else:
                    # case1. bottom Error
                    if 'bottom' in infer_result[first_idx:first_idx + end_idx]:
                        infer_result = door_closed.infer_bottom_check(crop_img_list_origin, infer_result, first_idx, end_idx, floor, cell)

                    # case2. model vision error
                    columns = cell+1
                    kakao_context = f'{floors}층 {columns}칸 비전 확인 요망\n'+\
                                    "(해당 디바이스에 없는 상품을 비젼으로 인식하여, 해당 셀의 POG로 변경 하였습니다.)"
                    send_slack = False
                    for index, contain in enumerate(infer_result[first_idx:first_idx + end_idx]):
                        if contain not in pog_list_by_design_infer_label:
                            pre_vision_label = ''
                            if index == 0 and compare_list_master[cell]['design_pkey_first'] is not None:
                                pre_vision_label = dao.get_designs_by_design_pkey(int(compare_list_master[cell]['design_pkey_first'])).design_infer_label 
                            elif index == 1 and compare_list_master[cell]['design_pkey_second'] is not None:
                                pre_vision_label = dao.get_designs_by_design_pkey(int(compare_list_master[cell]['design_pkey_second'])).design_infer_label
                            elif index == 2 and compare_list_master[cell]['design_pkey_third'] is not None:
                                pre_vision_label = dao.get_designs_by_design_pkey(int(compare_list_master[cell]['design_pkey_third'])).design_infer_label
                            else:
                                pre_vision_label = 'empty'
                            send_slack = True
                            kakao_context += f'\n{index}order\n'+\
                                            f'POG: {compare_list_master[cell]["design_infer_label"]}\n'+\
                                            f'비전 결과 : {infer_result[first_idx+index]}\n'+\
                                            f'이전 거래 비전 결과 : {pre_vision_label}\n'

                            log.warning(f'[({floor})floor/({cell})cell/({index})order]: 해당디바이스 POG에 해당하는 design_label이 존재하지 않습니다. [강제 pkey 변환 ({infer_result[first_idx+index]} -> {compare_list_master[cell]["design_infer_label"]})] ')
                            infer_result[first_idx+index] = compare_list_master[cell]['design_infer_label']
                    if send_slack:
                        door_closed.message.send_slack(trDate, kakao_context, devkind)
                log_label[floor][columns_i] = infer_result[first_idx:first_idx + end_idx]
                value[columns_i]['front_result'] = infer_result[first_idx:first_idx + end_idx]
                first_idx += end_idx

        
        log.info(f'---{time.time() - start_time} - bottom logic')

        ##비전결과 로그 찍기
        start_time = time.time()

        '''
            first_inference_result
        f0[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
        f1[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
            second_inference_result
        f0[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
        f1[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
            third_inference_result
        f0[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
        f1[`~~~,~~~,~~~,~~~,~~~,~~~,~~~,~~~]
        '''
        infer_result = {}
        infer_result[0] = []
        infer_result[1] = []
        infer_result[2] = []
        for floor, value in log_label.items():
            for columns_i, label in value.items():
                for i in range(result_of_front_num):
                    try:
                        infer_result[i].append(label[i])
                    except:
                        infer_result[i].append('empty')
                        
        tmp_log = ['first','second','third']
        for q,w in infer_result.items():
            log_str += f"------------{tmp_log[q]}_inference_result------------\n"
            for e,r in shelf_storage_dict.items():
                if r == 'NC':
                    log_str += f'{w[:8]}\n'
                    del w[:8]
                elif r == 'EC':
                    log_str += f'{w[:6]}\n'
                    del w[:6]
        util.LogGetter.log += log_str
        log.info(f'---{time.time() - start_time} - vision_inference_result log_write')


        #product 계산하는 곳 result_orderlist = [{'goodId' : '123456' , 'goodName' : 말보루 , 'RowNo' : 0 , 'ColNo':1 , 'goodscnt' : 3}, {...},{...},..]
        start_time = time.time()
        result['orderList'] = door_closed.product_count(final_result)
        log.info(f'---{time.time() - start_time} - product_count')

        #cigar_cell update
        start_time = time.time()
        for i in door_closed.cigar_cells_list:
            try:
                dao.update_cigar_cells(i)
            except Exception as err:
                log.error(traceback.format_exc())
                dao.session.rollback()
                raise Exception(f'[dao.update_cigar_cells] : {i} / {str(err)}')


        # cigar_stock delete 하고 insert 
        for cell_pkey, cigar_stock in door_closed.cnt_dict.items():
            dao.delete_cigar_stocks(cell_pkey)
            for design_pkey, stock_count in cigar_stock.items():
                dao.insert_cigar_stocks(cell_pkey, design_pkey, stock_count)
        log.info(f'---{time.time() - start_time} - update cigar_cell / cigar_stock')

        # cigar 재고 알림
        start_time = time.time()
        door_closed.check_mininum_stocks(result['orderList'])
        log.info(f'---{time.time() - start_time} - check mininum_stocks')
        dao.session.commit() 
    except Exception as e:
        log.error(traceback.format_exc())
        result['abort'] = {'code': 500, 'msg': f'{str(e)}'}
    finally:
        log.info(util.align_center(util.LogGetter.log))
        util.LogGetter.log = ''
        del door_closed
        dao.session.close()

    log.info(f'---{time.time() - tmp_start} - total_time')
    return json.dumps(result, ensure_ascii=False)

@cigar_bp.route('/')
@cigar_bp.route('/index')
def index():
    return devkind

# def align_center(log_str, max_len=100):
#     log_str = log_str.split('\n')
#     return '\n'.join([row.center(max_len, ' ') for row in log_str])

# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         traceback.print_exc()