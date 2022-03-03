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
from collections import Counter, defaultdict
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
import app.util as util
import threading

from glob import glob
import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#loop = asyncio.get_event_loop()

# Log 결정 Class
devkind = 'vaccine'
log = LogDesignate(devkind)

log.info(f'{os.getpid()}|{devkind}_inference| ########## S T A R T ##########')

vaccine_bp = Blueprint('main', __name__, url_prefix='/')

resize = (224, 224)
s3 = boto3.client('s3')

infer_Bucket_name = 'smart-retail-inference'
save_img_Bucket_name = 'smart-retail-server-log'
save_log_to_s3 = config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3
max_floor = 4

class Coordinate_refine:
    def __init__(self, companyId, storeId, deviceId, origin_coor):
        self.companyId = companyId
        self.storeId = storeId
        self.deviceId = deviceId
        self.origin_coor = origin_coor
        global s3
        self.s3 = s3
        self.dao = DAO()
        self.device_shelf_list = self.dao.get_device_shelf(self.companyId, self.storeId, self.deviceId)

    def draw_box(self, frame, result):
        for i in result:
            #print('box : ' , i)
            i = i[0]
            cv2.rectangle(frame, (int(i[0]),int(i[1])), (int(i[2]), int(i[3])), (0,0,255), 2)
            #cv2.rectangle(frame, (int(i[0]),int(i[1])), (int(i[2]), int(i[3])), (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 3)
            #cv2.putText(frame, calc_area(i), (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return frame

    def refine_coor(self, confidence = 0.6):
        final_coor = []
        for img_coor in self.origin_coor:
            tmp_coor = []
            for i in img_coor:
                coor = i[:4]
                coor = [int(i) for i in coor]
                conf = i[4]
                if conf < confidence:
                    continue
                label_num = i[5]
                tmp_coor.append([coor, conf, label_num])
            final_coor.append(tmp_coor)
        return final_coor

    def count(self, column_name : str, detection_result : list):
        pick_label = list(map(lambda x : x[2], detection_result))
        refined_dict = dict(Counter(pick_label))
        result = {column_name : {}}
        total_cnt = 0
        for i in refined_dict.keys():
            value = refined_dict[i]
            if i.split('_')[-1] == 'case':
                name = 'case'
                cnt = value * 10
            elif i.split('_')[-1] == 'halfcase':
                name = 'halfcase'
                cnt = value * 5
            elif i.split('_')[-1] == 'box':
                name = 'box'
                cnt = value * 1
            else:
                name = 'piece'
                cnt = value * 1
            total_cnt += cnt
            if name in result[column_name]:
                value = result[column_name][name] + value
            result[column_name][name] = value
        result[column_name]['total_cnt'] = total_cnt
        #result[column_name] = dict(result[column_name])
        return result

    def center_point(self, corr, xy = 'x'):
        if xy == 'x':
            # print('x : ', corr)
            min_corr, max_corr = corr[0], corr[2]
            center_point = (max_corr + min_corr) / 2
        elif xy == 'y':
            # print('y : ', corr)
            min_corr, max_corr = corr[1], corr[3]
            center_point = (max_corr + min_corr) / 2
        return int(center_point)

    def get_line(self, floor, camera):
        line_coor = {}
        line_path = f'inference/lines/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}'
        try:
            line_data = self.s3.get_object(Bucket=f'{infer_Bucket_name}', Key=f'{line_path}/line.xml')['Body']
            root = minidom.parse(line_data)
            line_r1 = root.getElementsByTagName('r1')
            xleft = int(line_r1[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_r1[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_r1[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_r1[0].childNodes[7].childNodes[0].nodeValue)
            r1 = yleft
            
            line_c1 = root.getElementsByTagName('c1')
            xleft = int(line_c1[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_c1[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_c1[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_c1[0].childNodes[7].childNodes[0].nodeValue)
            c1 = xleft

            line_c2 = root.getElementsByTagName('c2')
            xleft = int(line_c2[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_c2[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_c2[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_c2[0].childNodes[7].childNodes[0].nodeValue)
            c2 = xleft

            line_c3 = root.getElementsByTagName('c3')
            xleft = int(line_c3[0].childNodes[1].childNodes[0].nodeValue)
            yleft = int(line_c3[0].childNodes[3].childNodes[0].nodeValue)
            xright = int(line_c3[0].childNodes[5].childNodes[0].nodeValue)
            yright = int(line_c3[0].childNodes[7].childNodes[0].nodeValue)
            c3 = xleft
        except Exception as err:
            log.error(traceback.format_exc())
            log.info(f'{line_path}/line.xml 이 존재하지 않습니다. 기본 라인으로 설정합니다.\n{err}')
            r1 = 340
            c1 = 200
            c2 = 670
            c3 = 1500
        return r1, c1, c2, c3 

    def split_normal_section(self, detection_result, column_type, floor, camera):
        r1, c1, c2, c3 = self.get_line(floor, camera) 
        if column_type == 'split':   # 1, 2 컬럼
            column_1 = []
            column_2 = []
            for i in detection_result:
                coor = i[0]
                x_coor = self.center_point(coor, xy = 'x')
                y_coor = self.center_point(coor, xy = 'y')
                if y_coor > r1 and x_coor > c1 and x_coor < c2:
                    column_1.append(i)
                elif y_coor > r1 and x_coor > c2 and x_coor < c3:
                    column_2.append(i)
            return column_1, column_2
        else:  # merge - 3컬럼
            column = []
            for i in detection_result:
                coor = i[0]
                x_coor = self.center_point(coor, xy = 'x')
                y_coor = self.center_point(coor, xy = 'y')
                if y_coor > r1 and x_coor > c1 and x_coor < c3:
                    column.append(i)
            return column

    def final_calc(self, rst_list):
        total_dic = {}
        for index, device_shelf in self.device_shelf_list.iterrows():
            floor = f"{device_shelf['shelf_floor']}"
            get_cells_info = self.dao.get_cells_by_shelf_pkey(device_shelf['shelf_pkey'])
            total_column_list = []
            vaccine_cell_list = []
            tmp_dic = {}
            for idx, vaccine_cell in get_cells_info.iterrows():
                total_column_list.append(vaccine_cell['cell_column'])
                vaccine_cell_list.append(vaccine_cell)
            total_column_list.extend(total_column_list)
            
            for i, rst in zip(total_column_list, rst_list[int(floor)*len(total_column_list):]):
                col = i
                if col in tmp_dic:
                    tmp_dic[col] = dict(Counter(tmp_dic[col]) + Counter(self.count(col, rst)[col]))
                    if tmp_dic[col] == {}:
                        tmp_dic.update(self.count(col, rst))
                else:
                    tmp_dic.update(self.count(col, rst))

            for col, vaccine_cell in enumerate(vaccine_cell_list):
                tmp_dic[col]['design_pkey_master'] =  vaccine_cell['design_pkey_master']
                tmp_dic[col]['cell_pkey'] = vaccine_cell['cell_pkey']

            tmp_dic.update(tmp_dic)
            total_dic[f'{floor}'] = tmp_dic
        return total_dic

    def refine_column(self, result):
        refined_coor = []
        save_xml_img = []
        cameras = Config.CAMERAS_LOCATION['VACCINE']
        for floor in range(max_floor):
            for i, j in enumerate(result[floor * max_floor: (floor+1) * max_floor]): ## 컬럼 늘어나면 추후 변동 가능 db에서 뽑아올 것 
                if cameras[i] == 'fl' or cameras[i] == 'bl':
                    col_1, col_2 = self.split_normal_section(detection_result = j, ## v0, v2
                                            column_type = 'split', floor = floor, camera = cameras[i])
                    refined_coor.append(col_1)
                    refined_coor.append(col_2)
                    save_xml_img.append(col_1 + col_2)
                else:
                    col = self.split_normal_section(detection_result = j, ## v1, v3
                                            column_type = 'merge', floor = floor, camera = cameras[i])
                    refined_coor.append(col)
                    save_xml_img.append(col)
        return refined_coor, save_xml_img

class VaccineDoorClosed:
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
        self.message = Alarm(companyId, storeId, deviceId, work_user)
        self.pre_trDate = self.re.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_trDate')
        self.main_df = []
        self.device_shelf_list = self.dao.get_device_shelf(self.companyId, self.storeId, self.deviceId)
        global s3
        self.s3 = s3

    def save_img_s3_or_local(self, save_path, image):
        if save_log_to_s3 == True:
            self.s3.put_object(Body=cv2.imencode('.jpg', image)[1].tostring(), Bucket=save_img_Bucket_name, Key=save_path)
        else:
            log.info("save_img")
            cv2.imwrite(f'{save_path}', image)

    def get_img_list_worker(self, return_dict, floor, cameras):
        start_time = time.time()
        img_list = []
        for camera in cameras:
            saved_full_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}/{self.trDate.split("_")[0]}'
            util.createFolder(saved_full_path)
            img = self.re_img.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_f{floor}_cam{camera}')
            encoded_img = np.frombuffer(img, dtype=np.uint8) 
            image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 
            t1 = threading.Thread(target=self.save_img_s3_or_local, args=(f'{saved_full_path}/{self.trDate}.jpg', image,))
            t1.daemon = True 
            t1.start()

            img_2_byte = cv2.imencode('.png',image)[1].tobytes()
            img_list.append(('image',['image', img_2_byte, [0.3, 0.3]]))
        return_dict[floor] = img_list
    
    def get_img_list(self, ):
            start_time = time.time()
            final_img_list = []
            get_img_list_manager = multiprocessing.Manager()
            return_dict = get_img_list_manager.dict()
            get_img_list_jobs = []
            for index, device_shelf in self.device_shelf_list.iterrows():
                floor = device_shelf['shelf_floor']
                # saved_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'
                p = multiprocessing.Process(target=self.get_img_list_worker,args=(return_dict, floor, Config.CAMERAS_LOCATION['VACCINE']))
                get_img_list_jobs.append(p)
                p.start()
            for get_img_list_proc in get_img_list_jobs:
                get_img_list_proc.join()
            
            for img_key, img_value in sorted(return_dict.items()):
                final_img_list.extend(img_value)
            return final_img_list

    def product_count(self, final_result):
        orderlist = {}
        total_log = {}
        floors = list(final_result.keys())
        goods_log = [[] for f in floors]
        cnt_dict = {}
        stock_count = {}
        total_order = []
        '''
        bf_stock - af_stock 해서 orderlist 만드는 곳 
        update_vaccine_cell_list = []
        orderlist = {cell_pkey1 : {design_pkey_master: cnt}, 
                    cell_pkey2 : {design_pkey_master: cnt}
                    ...}
        '''
        for floor, value in final_result.items():
            for col, col_info in value.items():
                cell_pkey = col_info['cell_pkey']
                design_pkey_master = col_info['design_pkey_master'] 
                orderlist[cell_pkey] = {}
                bf_stock = self.dao.get_vaccine_stocks(cell_pkey)
                for i,j in bf_stock.items():
                    orderlist[cell_pkey][i] = j
                orderlist[cell_pkey][design_pkey_master] = bf_stock.get(design_pkey_master,0) - col_info['total_cnt']                


        '''
        cnt_dict에 pkey를 goods_id로 바꿔서 복사하기 trade_log 용
        '''
        for floor, value in final_result.items():
            for col, col_info in value.items():
                stock_count[col_info['cell_pkey']] = {}
                goods_id = self.dao.get_designs_by_design_pkey(col_info['design_pkey_master'])
                stock_count[col_info['cell_pkey']][goods_id.goods_id] = col_info['total_cnt']
        
        '''
        orderlist로 포맷맞추기 -> total_order
        total_order = [{'goodId': -- , 'goodName': -- , 'RowNo' : -- , 'ColNo': --, 'goodsCnt': --}, {...}, {...}, ...]
        '''
        for cell_pkey, r in orderlist.items():
            trade_goods_name_value = {}  # for log
            for pkey, cnt in r.items():
                if cnt != 0:
                    design_pkey = self.dao.get_designs_by_design_pkey(str(pkey))
                    floor_cell_data = self.dao.get_cell_column_shelf_floor_vaccine(cell_pkey)
                    goods_price = self.dao.get_sale_price(self.storeId, str(pkey))
                    total_order.append({'goodId':design_pkey.goods_id,
                                        'goodsName':self.dao.get_goods_name(design_pkey.goods_id),
                                        'goodsCnt':str(cnt),
                                        'goodsPrice':None if goods_price is None else str(goods_price),
                                        'RowNo':floor_cell_data['shelf_floor'],
                                        'ColNo':floor_cell_data['cell_column'],
                                        'design_infer_label': design_pkey.design_infer_label
                                    })
                    
                    # db 로그테이블 insert
                    self.log_dao.insert_vaccine_trade_log(
                        vaccine_trade_log_no = self.trNo,
                        vaccine_trade_log_date = self.trade_date,
                        vaccine_trade_log_time = self.trade_time,
                        company_id = self.companyId,
                        store_id = self.storeId,
                        device_id = self.deviceId,
                        shelf_floor = floor_cell_data['shelf_floor'],
                        cell_column = floor_cell_data['cell_column'],
                        goods_id = design_pkey.goods_id,
                        goods_name = self.dao.get_goods_name(design_pkey.goods_id),
                        goods_label = design_pkey.design_infer_label,
                        goods_count = str(cnt),
                        stock_left = str(stock_count[cell_pkey][design_pkey.goods_id]),
                        duration = None,
                        work_user = self.work_user,
                        work_type = "trade",
                        status_code = None,
                        total_cnt = stock_count[cell_pkey][design_pkey.goods_id],
                        sale_price = None,
                        total_sale_price = None,
                    )
                    self.log_dao.session.commit()
                    trade_goods_name_value[self.dao.get_goods_name(design_pkey.goods_id)] = trade_goods_name_value.get(self.dao.get_goods_name(design_pkey.goods_id), 0) + cnt
            goods_log[self.dao.get_cell_column_shelf_floor_vaccine(cell_pkey)['shelf_floor']].append(trade_goods_name_value)
        try:
            '''
            update vaccine_cells 
            '''
            for floor, value in final_result.items():
                for col, col_info in value.items():
                    update_vaccine_cell_dict = {}
                    update_vaccine_cell_dict.setdefault('cell_pkey', None)
                    update_vaccine_cell_dict.setdefault('total_cnt', None)
                    update_vaccine_cell_dict['cell_pkey'] = col_info['cell_pkey']
                    update_vaccine_cell_dict['total_cnt'] = col_info['total_cnt']
                    try:
                        self.dao.update_vaccine_cells(update_vaccine_cell_dict)
                    except Exception as err:
                        log.error(traceback.format_exc())
                        raise Exception(f'[dao.update_vaccine_cells] : {update_vaccine_cell_dict} || {str(err)}')
            '''
            delete / insert vaccine_stocks
            '''
            for floor, value in final_result.items():
                for col, col_info in value.items():
                    try:
                        self.dao.delete_vaccine_stocks(col_info['cell_pkey'])
                        self.dao.insert_vaccine_stocks(col_info['cell_pkey'], col_info['design_pkey_master'], col_info['total_cnt'])
                    except Exception as err:
                        log.error(traceback.format_exc())
                        raise Exception(f'[delete / insert vaccine_stocks] : {floor}f | {col_info} || {str(err)}')
            '''
            trade_check_insert
            '''
            total_price = 0
            for check_pog in total_order:
                total_price += int(check_pog['goodsCnt']) * int(check_pog['goodsPrice'])
            try:
                try:
                    qr_confirm= self.re.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_qr_confirm')
                    qr_confirm = eval(qr_confirm)
                except Exception as err:
                    log.warning(traceback.format_exc())
                    log.warning(f'{self.companyId}_{self.storeId}_{self.deviceId}_qr_confirm | redis get error {err}')
                
                self.dao.insert_vaccine_trade_check(
                    vaccine_trade_date = self.trade_date,
                    vaccine_trade_time = self.trade_time,
                    company_id = self.companyId,
                    store_id = self.storeId,
                    device_id = self.deviceId,
                    total_sale_price = total_price,
                    qr_data = str(qr_confirm["qr_code"]) if qr_confirm else None,
                    user_level = str(qr_confirm["userLevel"]) if qr_confirm else None
                )
                self.dao.session.commit()
            except Exception as err:
                log.error(traceback.format_exc())
                raise Exception(f'[trade_check_insert] : {self.companyId}_{self.storeId}_{self.deviceId}_{total_price}_{qr_data}')
            '''
            trade_check_sub_insert
            '''
            check_pkey = self.dao.get_vaccine_trade_check_pkey(
                            self.trade_date,
                            self.trade_time,
                            self.companyId,
                            self.storeId,
                            self.deviceId
                        )
            try:
                for idx, check_pog in enumerate(total_order):
                    self.dao.insert_vaccine_trade_pog(
                        vaccine_trade_check_pkey = check_pkey,
                        vaccine_trade_no = idx,
                        shelf_floor = int(check_pog['RowNo']),
                        cell_column = int(check_pog['ColNo']),
                        goods_id = check_pog['goodId'],
                        goods_name = check_pog['goodsName'],
                        goods_label = check_pog['design_infer_label'],
                        goods_count = int(check_pog['goodsCnt']),
                        sale_price = int(check_pog['goodsPrice'])
                    )
                    check_pog.pop('design_infer_label')
            except Exception as err:
                log.error(traceback.format_exc())
                raise Exception(f'[trade_check_sub_insert] : {check_pog}')

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
            log_str = """
    ------------ goods -------------
    """ + "\n".join(list(map(lambda row: " | ".join([f"#{i} "+f"{cell if cell else ''}".center(8, " ") for (i, cell) in enumerate(row)]), goods_log))) + f"""

    ------------ total -------------
    {total_log_d if total_log_d else 'nothing in & out'}

    ---------- order list ----------
    {total_order if total_order else 'no orderlist'}
    """
            util.LogGetter.log += log_str

            ## 기본 로그들 S3에 저장
            all_floor_save_img_path = {}
            for idx, device_shelf in self.device_shelf_list.iterrows():
                floor_save_img_path = []
                floor = device_shelf['shelf_floor']
                cameras = Config.CAMERAS_LOCATION['VACCINE']
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
                'floors' : str(len(self.device_shelf_list.keys())),
                'request.path' : request.path,
                'request.url' : request.url,
                'request.remote_addr' : request.remote_addr,
                'request.method' : request.method,
                'request.data' : json.loads(request.data),
                'payment_error' : "False",
                'all_floor_save_img_path' : all_floor_save_img_path,
                'count_log' : util.align_center(util.LogGetter.log),
            }
            save_admin_json_path = f'logs/log/payment/{self.companyId}/{self.storeId}/{self.deviceId}/{self.trDate.split("_")[0]}'
            if save_log_to_s3 == True:
                self.s3.put_object(Body=bytes(json.dumps(admin_json_text, indent=4, ensure_ascii=False).encode('UTF-8')),
                                        Bucket=f'{save_img_Bucket_name}', Key=f'{save_admin_json_path}/{self.trDate.split("_")[-1]}.txt')
            else:
                if not os.path.exists(save_admin_json_path):
                    os.makedirs(save_admin_json_path)
                with open(f'{save_admin_json_path}/{self.trDate.split("_")[-1]}.txt', 'wt') as json_file:
                    json.dump(admin_json_text, json_file, indent=4)

        except Exception as e:
            log.error(f'ERROR -- {e}')
            log.error(traceback.format_exc())
            self.dao.session.rollback()   
        else:
            self.dao.session.commit()
        return total_order
    # full_saved_img_path = f'{saved_img_path}/{camera}/{trDate.split("_")[0]}'
    def save_img_xml(self, img_list, xml_list):
        for floor in range(max_floor):
            saved_img_path = f'logs/saved_box_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'
            util.createFolder(saved_img_path)
            for camera, img, xml in zip(Config.CAMERAS_LOCATION['VACCINE'], img_list[floor*max_floor:(floor + 1)*max_floor],xml_list[floor*max_floor:(floor + 1)*max_floor]):
                full_saved_img_path = f'{saved_img_path}/{camera}/{self.trDate.split("_")[0]}'
                encoded_img = np.frombuffer(img[1][1], dtype=np.uint8)
                image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                for i in xml:
                    tmp = i[0]
                    text = i[2]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image,text,(tmp[0],tmp[1]), font, 2, (0,0,255), 3)
                    cv2.rectangle(image, (tmp[0], tmp[1]), (tmp[2], tmp[3]), (0,0,255), 3)
                util.createFolder(full_saved_img_path)
                self.save_img_s3_or_local(f'{full_saved_img_path}/{self.trDate}.jpg', image)


@vaccine_bp.route('/vaccine_model', methods=['POST'])
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
        log_str += f'piece = 1 | each_case = 10 piece | each_box = 1 piece\n'
        util.LogGetter.log = log_str
        # Db class
        dao = DAO()
        ## 진호 시작
        
        start_time = time.time()
        door_closed = VaccineDoorClosed(companyId, storeId, deviceId, work_user,
                                       trDate, trade_date, trade_time, trNo, trResponseDate)
        start_time = time.time()
        
        # redis에서 전체 층의 이미지를 리스트로 가져옴 -> (멀티프로세싱)
        # s3 or save_local -> (스레딩 처리)
        try:
            infer_img_list = door_closed.get_img_list()
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[get_img_list] : {str(err)}')
        log.info(f'---{time.time() - start_time}- door_closed.get_img_list')

        # request to object detection server
        start_time = time.time()
        try:
            api_ip = 'https://125.132.250.227/predict/vaccine'
            api_key = 'eUAg9RfZ5AtoUejUr6uquTtFCeZ7vBpf9sxoNgzM'
            response = requests.post(api_ip, files=infer_img_list, headers={"X-API-KEY": api_key}, verify=False)
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[obj_request_to {api_ip}] : {str(err)}')
        log.info(f'---{time.time() - start_time}- requests.post')

        # 좌표 정제
        start_time = time.time()
        try:
            origin_coor = response.json()['res']
            refine = Coordinate_refine(companyId, storeId, deviceId, origin_coor)
            result_coor = refine.refine_coor()
            
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[refine.refine_coor] : {str(err)}')
        log.info(f'---{time.time() - start_time}- refine_coor')
        
        # 컬럼별로 결과값 대입
        start_time = time.time()
        try:
            refined_coor, save_xml_img = refine.refine_column(result_coor)
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[refine_column] : {str(err)}')
        log.info(f'---{time.time() - start_time}- refine_column')

        # 결과값 같은것끼리 묶기
        start_time = time.time()
        try:
            final_calc_dict = refine.final_calc(rst_list = refined_coor)
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[final_calc] : {str(err)}')
        log.info(f'---{time.time() - start_time}- final_calc')

        # xml _ img save
        start_time = time.time()
        try:
            t1 = threading.Thread(target=door_closed.save_img_xml, args=(infer_img_list, save_xml_img))
            t1.daemon = True 
            t1.start()
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[save_xml_img_thread] : {str(err)}')
        log.info(f'---{time.time() - start_time}- thread_save_xml&img')


        '''
        refine.final_calc format = a
        {'0': {'0': {'piece': 6, 'total_cnt': 26, 'case': 2, 'cell_pkey' : }, 
             '1': {'piece': 6, 'total_cnt': 10, 'box': 4}, 
             '2': {'piece': 20, 'total_cnt': 20}}, 
        '1': {'0': {'piece': 6, 'total_cnt': 26, 'case': 2}, 
             '1': {'piece': 6, 'total_cnt': 10, 'box': 4},
             '2': {'piece': 20, 'total_cnt': 20}}, 
        '2': {'0': {'piece': 6, 'total_cnt': 26, 'case': 2}, 
             '1': {'piece': 6, 'total_cnt': 10, 'box': 4}, 
             '2': {'piece': 20, 'total_cnt': 20}}, 
        '3': {'0': {'piece': 6, 'total_cnt': 26, 'case': 2}, 
             '1': {'piece': 6, 'total_cnt': 10, 'box': 4}, 
             '2': {'piece': 20, 'total_cnt': 20}}}
        '''
        if isinstance(final_calc_dict,dict):
            for key1, value1 in final_calc_dict.items():
                util.LogGetter.log += f'-------------------{key1} floor-------------------\n'
                for key2, value2 in value1.items():
                    util.LogGetter.log += f'{key2}# | {value2}\n'
                    
        # product count orderlist 뽑기
        start_time = time.time()
        try:
            result['orderList'] = door_closed.product_count(final_calc_dict)
        except Exception as err:
            log.error(traceback.format_exc())
            raise Exception(f'[product_count] : {str(err)}')
        log.info(f'---{time.time() - start_time}- product_count')
        log.info(f'---{time.time() - tmp_start} - total_time')
        door_closed.message.send_slack(trDate,f':smile:문닫힘 SUCCESS:smile:',devkind)
    except Exception as e:
        log.error(f'[vaccine] <tf_model> Error ({str(e)})')
        log.error(traceback.format_exc())
        door_closed.message.send_slack(trDate,f':face_with_symbols_on_mouth:문닫힘 ERROR:face_with_symbols_on_mouth:\n{e}',devkind)
        result['abort'] = {'code': 500, 'msg': f'{str(e)}'}
    finally:
        log.info(util.align_center(util.LogGetter.log))
        dao.session.close()
        return json.dumps(result, ensure_ascii=False)


@vaccine_bp.route('/')
@vaccine_bp.route('/index')
def index():
    return devkind
