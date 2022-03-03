# -*- coding:utf-8 -*-
from operator import is_
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
import threading

from app import models
import traceback
import copy
import logging.config
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import multiprocessing
import boto3

from keys import keys
from PIL import Image
from flask_cors import CORS
from .data_access import DAO
from datetime import datetime
from xml.dom import minidom

import app.log_adapter
import requests

from tensorflow.keras.models import load_model
from tensorflow import make_tensor_proto, make_ndarray
from flask import Flask, request, render_template, redirect, Blueprint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from .config import Config, config_by_name
from .alarm import Alarm
from ec2_metadata import ec2_metadata

from . import shortall
from .lc_inf import OrderList
# from .log_getter import LogGetter
from .log_designate import LogDesignate
import app.util as util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loop = asyncio.get_event_loop()

# Log 결정 Class
devkind = 'fridge'
log = LogDesignate(devkind)

log.info(f'{os.getpid()}|{devkind}_inference| ########## S T A R T ##########')

fridge_bp = Blueprint('main', __name__, url_prefix='/')

# 냉장고 doorclosed Class
class FridgeDoorClosed:
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
        
        try:
            self.re = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
                                db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
                                password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD, charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
                                decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)
        except Exception as err:
            log.info(f'self.re 에러 {err}')    
        try:
            self.dao = DAO()
        except Exception as err:
            log.info(f'self.dao 에러 {err}')

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
        self.save_xml_path_dict = None

        try:
            self.s3 = boto3.client('s3')
        except Exception as err:
            log.info(f'boto3.client{err}')
        util.LogGetter.log = ''
        util.LogGetter.log += f"""
                {'- Inference info -'.center(100, ' ')}
                {f'companyId:{self.companyId}|storeId:{self.storeId}|deviceId:{self.deviceId}'.center(100, ' ')}
                {f'trNo:{self.trNo}|trDate:{self.trDate}|work_user:{self.work_user}|alarm:{self.message.device_alarm}'.center(100, ' ')}
                """

    def __del__(self):
        self.dao.session.close()
    
    def set_save_xml_path(self, save_xml_path_dict):
        self.save_xml_path_dict = save_xml_path_dict

    def product_count(self, infer_work_flag):
        try:
            '''
                문을 닫고 상품 어떤 물건이 몇개 빠졌는지 확인하는 로직
            '''
            # load cell
            result = {}
            start_time = time.time()
            start = time.time()
            try:
                ol = OrderList(self.companyId, self.storeId, self.deviceId, self.trNo, self.trade_date, self.trade_time, infer_work_flag, self.work_user)
                log.info(f'---{time.time() - start_time} - ol객체생성()')
            except Exception as err:
                log.info(f'orderlist 객체 생성 {err}')
            start_time = time.time()
            alert_dict, result, count_log, ol_master = ol.count(self.work_user)
            log.info(f'---{time.time() - start_time} - ol.count()')
            
            count_log = f'{self.count_vision_log}\n{count_log}'
            alert = False
            alert_list = []
            cell_alert_list = []
            error_point = []
            payment_error4client = []
            all_floor_save_img_path = {}


            alarm_msg = ""
            for alert_key, alert_value in alert_dict.items():
                cell_list = []
                floor_save_img_path = []
                pre_floor_save_img_path = []

                if self.save_log_to_s3 == True:
                    s3_list_objects = ((self.s3.list_objects_v2(
                        Bucket=f'{self.infer_Bucket_name}',
                        Prefix =f'inference/boxes/{self.companyId}/{self.storeId}/{self.deviceId}/{alert_key}/', Delimiter=f'/')['CommonPrefixes']))
                    cameras = list(map(lambda x : os.path.basename(os.path.normpath(x['Prefix'])), s3_list_objects))
                else:
                    cameras = sorted(os.listdir(f'inference/boxes/{self.companyId}/{self.storeId}/{self.deviceId}/{alert_key}'))

                for camera in cameras:
                    floor_save_img_path.append(f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{alert_key}/{camera}/{self.trDate.split("_")[0]}/{self.trDate}.jpg')
                    if self.pre_trDate is not None:
                        pre_floor_save_img_path.append(f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{alert_key}/{camera}/{self.pre_trDate.split("_")[0]}/{self.pre_trDate}.jpg')

                all_floor_save_img_path[str(alert_key)] = floor_save_img_path

                if 1 in alert_value.values():
                    alert = True
                    floor_error_point = {}
                    log.error(f'[LC]c:{self.companyId}/s:{self.storeId}/d:{self.deviceId}/f:{alert_key}/res:{list(alert_value.values())}')
                    # admin log
                    floor_error_point["floor"] = alert_key
                    floor_error_point["save_img_path"] = floor_save_img_path
                    floor_error_point["pre_save_img_path"] = pre_floor_save_img_path
                    floor_error_point["error_cells"] = []

                    alert_list.append(alert_key)
                    for cell, error in alert_value.items():
                        if error == 1:
                            cell_list.append(int(cell)+1)
                            column_info = {}
                            # column_info["cell"] = int(cell)

                            ol_column = ol_master.query(f'shelf_floor == {int(alert_key)} & cell_column == {int(cell)}')
                            column_info["shelf_floor"] = ol_column['shelf_floor'].tolist()[0]
                            column_info["cell_column"] = ol_column['cell_column'].tolist()[0]
                            column_info["weight_open"] = ol_column['w_o'].tolist()[0]
                            column_info["weight_close"] = ol_column['w_c'].tolist()[0]

                            design_pkey_master = ol_column['design_pkey_master'].tolist()[0]
                            dpm_goods_rst = self.dao.get_goods_by_design_pkey(design_pkey_master)
                            dpm_design_rst = self.dao.get_designs_by_design_pkey(design_pkey_master)
                            column_info["design_pkey_master"] = \
                                {'pkey' : design_pkey_master, 'barcode':dpm_goods_rst.goods_id ,
                                 'goods_name' : dpm_goods_rst.goods_name, 'infer_label' : dpm_goods_rst.design_infer_label,
                                 'weight' : dpm_design_rst.design_mean_weight}

                            column_info["inference_mode"] = ol_column['inference_mode'].tolist()[0]
                            column_info["load_cell_mode"] = ol_column['load_cell_mode'].tolist()[0]
                            column_info["stock_count_max"] = ol_column['stock_count_max'].tolist()[0]
                            stock = ol_column['stock'].tolist()[0]
                            stock_keys = list(stock.keys())
                            for stock_key in stock_keys:
                                stock_goods_rst = self.dao.get_goods_by_design_pkey(stock_key)
                                stock[f'{stock_goods_rst.goods_name}({stock_goods_rst.goods_id})'] = stock.pop(stock_key)
                            column_info["stock"] = stock

                            #
                            lc_inf_weight = ol_column['lc_inf_weight'].tolist()[0]
                            lc_inf_count = ol_column['lc_inf_count'].tolist()[0]
                            lc_inf_permit_rate = ol_column['lc_inf_permit_rate'].tolist()[0]
                            lc_inf_permit_else_rate = ol_column['lc_inf_permit_else_rate'].tolist()[0]
                            column_info["lc_inf_count"] = lc_inf_count
                            column_info["lc_inf_permit_rate"] = lc_inf_permit_rate
                            column_info["lc_inf_permit_else_rate"] = lc_inf_permit_else_rate
                            column_info["lc_inf_weight"] = lc_inf_weight

                            lc_inf_count_minus = int(lc_inf_count) - 1
                            lc_inf_count_plus = int(lc_inf_count) + 1

                            error_weight_range = {}
                            for admin_lc_count in [lc_inf_count_minus, lc_inf_count, lc_inf_count_plus]:
                                permit = lc_inf_permit_rate.get(admin_lc_count, lc_inf_permit_else_rate)
                                control_limit = dpm_design_rst.design_mean_weight * permit
                                limit_min = (dpm_design_rst.design_mean_weight * admin_lc_count) - control_limit
                                limit_max = (dpm_design_rst.design_mean_weight * admin_lc_count) + control_limit
                                error_weight_range[admin_lc_count] = [limit_min, limit_max]
                            column_info["error_weight_range"] = error_weight_range

                            design_pkey_front = ol_column['design_pkey_front'].tolist()[0]
                            # if design_pkey_front == None:
                            if pd.isnull(design_pkey_front):
                                column_info["pre_design_pkey_front"] = f'empty'
                            else:
                                pdpf_goods_rst = self.dao.get_goods_by_design_pkey(design_pkey_front)
                                pdpf_design_rst = self.dao.get_designs_by_design_pkey(design_pkey_front)
                                column_info["pre_design_pkey_front"] = \
                                    {'pkey' : design_pkey_front, 'goods_name' : pdpf_goods_rst.goods_name,
                                     'infer_label' : pdpf_goods_rst.design_infer_label, 'weight' : pdpf_design_rst.design_mean_weight}

                            design_pkey_inf_main = ol_column['design_pkey_inf_main'].tolist()[0]
                            design_pkey_inf_empty = ol_column['design_pkey_inf_empty'].tolist()[0]
                            if design_pkey_inf_empty == 0:
                                column_info["design_pkey_front"] = 'empty'
                            else:
                                dpf_goods_rst = self.dao.get_goods_by_design_pkey(design_pkey_inf_main)
                                dpf_design_rst = self.dao.get_designs_by_design_pkey(design_pkey_inf_main)
                                column_info["design_pkey_front"] = \
                                    {'pkey' : design_pkey_inf_main, 'goods_name' : dpf_goods_rst.goods_name,
                                     'infer_label' : dpf_goods_rst.design_infer_label, 'weight' : dpf_design_rst.design_mean_weight}


                            # design_pkey_front_c = ol_column['design_pkey_front_c'].tolist()[0]
                            # if design_pkey_front_c == None:
                            # if pd.isnull(design_pkey_front_c):
                            #     column_info["design_pkey_front"] = f'empty'
                            # else:
                            #     dpf_goods_rst = self.dao.get_goods_by_design_pkey(design_pkey_front_c)
                            #     dpf_design_rst = self.dao.get_designs_by_design_pkey(design_pkey_front_c)
                            #     column_info["design_pkey_front"] = {'pkey' : design_pkey_front_c, 'goods_name' : dpf_goods_rst.goods_name, 'infer_label' : dpf_goods_rst.design_infer_label, 'weight' : dpf_design_rst.design_mean_weight}

                            floor_error_point["error_cells"].append(column_info)

                            # general log
                            payment_error4client.append({
                                "error_type": "scope",
                                "floor": str(alert_key),
                                "img_path": floor_save_img_path,
                                "cell_alert": str(cell),
                                "design_goods_name" : f'{dpm_goods_rst.goods_name}({dpm_goods_rst.goods_id})',
                                "pre_save_img_path" : pre_floor_save_img_path
                            })

                    error_point.append(floor_error_point)
                if len(cell_list):
                    cell_alert_list.append(cell_list)
                    device_pkey = self.dao.get_device_pkey(self.companyId, self.storeId, self.deviceId)
                    reverse_floor = self.dao.get_max_floor(device_pkey) - int(alert_key) + 1
                    for cell_alert in cell_list:
                        goods_name = self.dao.get_goods_name_cell_pkey(device_pkey, alert_key, int(cell_alert)-1)
                        alarm_msg += f"- {reverse_floor}층 {cell_alert}칸 {goods_name[0]} 자리\n\n"

            # set redis trDate
            self.re.set(f'{self.companyId}_{self.storeId}_{self.deviceId}_trDate', self.trDate)

            # 기본 로그들 S3에 저장
            admin_json_text = {
                'trNo' : self.trNo,
                'trade_date' : str(self.trade_date),
                'trade_time' : str(self.trade_time),
                'companyId' : self.companyId,
                'storeId' : self.storeId,
                'deviceId' : self.deviceId,
                'floors' : str(len(alert_dict)),
                'request.path' : request.path,
                'request.url' : request.url,
                'request.remote_addr' : request.remote_addr,
                'request.method' : request.method,
                'request.data' : json.loads(request.data),
                'payment_error' : "True" if alert else "False",
                'all_floor_save_img_path' : all_floor_save_img_path,
                'all_floor_save_xml_path' : self.save_xml_path_dict["all_floor_save_xml_path"],
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

            if alert:
                admin_error_json_text = {
                    'trNo' : self.trNo,
                    'trade_date' : str(self.trade_date),
                    'trade_time' : str(self.trade_time),
                    'companyId' : self.companyId,
                    'storeId' : self.storeId,
                    'deviceId' : self.deviceId,
                    'floors' : str(len(alert_dict)),
                    'request.path' : request.path,
                    'request.url' : request.url,
                    'request.remote_addr' : request.remote_addr,
                    'request.method' : request.method,
                    'request.data' : json.loads(request.data),
                    'count_log' : count_log,
                    'error_floors': error_point
                }

                save_json_path = f'logs/error_log/{self.companyId}/{self.storeId}/{self.deviceId}/{self.trDate.split("_")[0]}'
                save_admin_error_json_path = f'logs/error_log_admin/payment/{self.companyId}/{self.storeId}/{self.deviceId}/{self.trDate.split("_")[0]}'
                if self.save_log_to_s3 == True:
                    self.s3.put_object(Body=bytes(json.dumps(payment_error4client, indent=4, ensure_ascii=False).encode('UTF-8')),
                                       Bucket=f'{self.save_img_Bucket_name}', Key=f'{save_json_path}/{self.trDate.split("_")[-1]}.txt')
                    self.s3.put_object(Body=bytes(json.dumps(admin_error_json_text, indent=4, ensure_ascii=False).encode('UTF-8')),
                                       Bucket=f'{self.save_img_Bucket_name}', Key=f'{save_admin_error_json_path}/{self.trDate.split("_")[-1]}.txt')

                else:

                    if not os.path.exists(save_json_path):
                        os.makedirs(save_json_path)
                    if not os.path.exists(save_admin_error_json_path):
                        os.makedirs(save_admin_error_json_path)
                    with open(f'{save_json_path}/{self.trDate.split("_")[-1]}.txt', 'wt') as json_file:
                        json.dump(payment_error4client, json_file, indent=4)
                    with open(f'{save_admin_error_json_path}/{self.trDate.split("_")[-1]}.txt', 'wt') as json_file:
                        json.dump(admin_error_json_text, json_file, indent=4)

                if self.conf_email_alarm == True and alarm_msg != "":
                    if self.work_user == 'manager':
                        header = "(관리자 모드)"
                        send_mode = "default"
                    elif self.work_user == 'interminds':
                        header = "(테스트 모드)"
                        send_mode = "interminds"
                    else:
                        header = "(사용자 모드)"
                        send_mode = "default"

                    try:
                        self.message.send_alimtalk("scope_error", self.trDate, alarm_msg, send_mode, header)
                    except Exception as exx:  # 카톡 알림 발송 오류
                        traceback.print_exc()
                        log.error(f'[kakaotalk] error {exx}')

                result['abort'] = {'code': 600, 'msg': f'LoadCell Error({str(alarm_msg).strip()})'}  # 600 Server Error
            else:
                ol.dao.session.commit() # lc_inf (OrderList commit) / 재고수 등 [insert/update]
                self.check_mininum_stocks(result['orderList'])
                result['abort'] = {'code': 200, 'msg': ''}   # 200 정상

            util.LogGetter.log += '\n' + f'lc inf: {time.time() - start}'.center(100)
            result['trDate'] = self.trResponseDate
            result['trNo'] = self.trNo
        except Exception as e:
            log.error(traceback.format_exc())
            log.error(f'[product_count] Error ({str(e)})')
            result['abort'] = {'code': 601, 'msg': f'product_count Error({str(e)})'}  # 601 Exception Error
        finally:
            """
                - OrderList 클래스 객체 삭제 -
                
                def __del__(self):
                    self.dao.session.close()
                    self.log_dao.session.close()
            """
            del ol

        return result

    def check_mininum_stocks(self, order_list):
        '''
            셀의 재고가 알람 개수 이하면 카톡 알림
            (재고가 알람 개수 초과에서 이하로 변경된 경우, 재고 모드가 abs인 경우, 무게 오류 발생하지 않은 경우만 알림)
            섞인 상품은 제외
        '''

        # get minium stock
        # ("층", "칸", "바코드", "상품명", "현재 재고", "알림 개수")
        stock_list = self.dao.get_stocks_by_csd_id(self.companyId, self.storeId, self.deviceId)

        # if not exist minium stock -> return
        if len(stock_list) == 0 or len(order_list) == 0:
            return

        # need reverse floor
        device_pkey = self.dao.get_device_pkey(self.companyId, self.storeId, self.deviceId)
        floor = self.dao.get_max_floor(device_pkey)

        alarm_msg = ""
        for line in stock_list:
            line = list(line)
            alarm_check = [ol for ol in order_list if int(ol['RowNo'])==line[0] and int(ol['ColNo'])==line[1] and int(ol['goodsCnt'])+line[4] > line[5]]
            if alarm_check :
                line[0] = floor-line[0]+1
                line[1] += 1
                alarm_msg += "- {0}층 {1}칸 {3}({2}) {4}개 ({5}개 이하 알림)\n\n".format(*line)
        if self.conf_email_alarm == True and alarm_msg != "":
            if self.work_user == 'manager':
                header = "(관리자 모드)"
                send_mode = "default"
            elif self.work_user == 'interminds':
                header = "(테스트 모드)"
                send_mode = "interminds"
            else:
                header = "(사용자 모드)"
                send_mode = "default"
            try:
                self.message.send_alimtalk("stock_alarm", self.trDate, alarm_msg, send_mode, header)
            except Exception as exx:  # 카톡 알림 발송 오류
                log.error(f'[kakaotalk] error {exx}')
                log.error(traceback.format_exc())

# 냉장고 Inference Class
class FridgeInference:
    def __init__(self, companyId, storeId, deviceId, trDate, re, header, work_user):
        self.companyId = companyId
        self.storeId = storeId
        self.deviceId = deviceId
        self.trDate = trDate
        self.str_ymd = str(trDate).split('_')[0]
        self.str_hms = str(trDate).split('_')[1]
        self.save_xml_path_dict = {}
        self.save_xml_path_dict["all_floor_save_xml_path"] = {}
        try:
            self.dao = DAO()
        except Exception as err:
            log.info(f'self.dao 에러{err}')
        self.res = {}
        self.res['abort'] = {'code': 200, 'msg': ''}  # 200
        self.log_str = ""
        self.infer_Bucket_name = 'smart-retail-inference'
        self.save_img_Bucket_name = 'smart-retail-server-log'
        self.save_log_to_s3 = config_by_name[Config.BOILERPLATE_ENV].SAVE_LOG_TO_S3
        self.conf_email_alarm = config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM
        self.resize = (224, 224)
        self.re = re
        self.header = header
        self.message = Alarm(companyId, storeId, deviceId, work_user)
        try:
            self.re_img = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
                                    db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
                                    password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD)
        except Exception as err:
            log.info(f're_img redis error{err}')
        try:
            self.s3 = boto3.client('s3')
        except Exception as err:
            log.info(f'boto3.client(){err}')
            
    def __del__(self):
        self.dao.session.close()

    def refine_coor(self, results):
        coors = list(map(lambda x : x[:4], results))
        final_coor = sorted([list(map(int, coor)) for coor in coors])
        return final_coor

    def get_short_tall_tag(self, boxes, line_path, inference_boxes):
        path_f = line_path.split('/')[-2]
        path_cam = line_path.split('/')[-1]
        line_path = line_path + '/line.xml'

        # short_tall_box = shortall.Box(boxes)
        # short_tall_box = shortall.BackFront(short_tall_box, detector)
        # short_tall_box = shortall.ShortTall(short_tall_box, seperator)
        result = ['empty']*4

        top_line, btm_line = self.get_lines(line_path)
        detector = shortall.get_streight(btm_line)
        seperator = shortall.get_streight(top_line)
        short_tall_box = shortall.Box(boxes, inference_boxes, detector, seperator)
    
        for box in boxes:
            iou_list = []
            for inference_box in inference_boxes:
                ttmp = shortall.iou(box,inference_box)
                iou_list.append(ttmp)
            max_index = np.argmax(iou_list)
            if short_tall_box.backfront(box):
                if result[max_index] != 'empty':
                    log.info(f'error {path_f}층 {path_cam}숏톨 박스 에러')
                    pass
                else:
                    result[max_index] = short_tall_box.short_tall(box)
        return result


    def fill_empty_part(self, item_list, empty_list):
        ##items_list <- short_tall_tags
        ##['#short', '#short', '#short', '#short', '#short', '#short', '#short', '#short', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall', '#tall']

        ##empty_list
        ##['product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product', 'product']
        items = copy.deepcopy(item_list)
        result = []
        for product_empty, st in zip(empty_list,items):
            if product_empty == 'empty':
                result.append('empty')
            else:
                result.append(st)
        return result
        

    def get_short_tall_pkeys_and_labels(self, df, labels, *tags):
        result_pkeys = []
        result_labels = []
        for i in range(len(labels)):
            # select by label
            label_res = df.query(f'design_infer_label == "{labels[i]}"').dropna()
            label_res = label_res.query('tag_value.str.contains("@")')['tag_value'].to_list()

            # select by '@'
            at_res = []
            for at in label_res:
                at_res = df.query(f'tag_value == "{at}"')['design_infer_label'].to_list()

            # select by tag
            tag_query = f'design_infer_label.isin({at_res}) & tag_value == "{tags[i]}"'
            query_res = df.query(tag_query)
            pkey = query_res['design_pkey'].to_list()
            label = query_res['design_infer_label'].to_list()

            # if not select, give pkey just by label
            if not pkey:
                query_res = df.query(f'design_infer_label == "{labels[i]}"')
                result_pkeys.append(query_res['design_pkey'].to_list()[0])
                if tags[i] == "empty":
                    result_labels.append("empty")
                else:
                    result_labels.append(query_res['design_infer_label'].to_list()[0])
            else: 
                result_pkeys.append(pkey[0])
                if tags[i] == "empty":
                    result_labels.append("empty")
                else:
                    result_labels.append(label[0])
                    
        return result_pkeys, result_labels

    def get_s3_objects(self, inference_dir, use_flag):

        s3_flag = True
        try:
            """
            # s3_list_objects : 
            [
                {'Prefix': 'inference/boxes/0001/00888/s_00008/2/l/'}, 
                {'Prefix': 'inference/boxes/0001/00888/s_00008/2/r/'}
            ]
            """
            s3_list_objects = ((self.s3.list_objects_v2(
                Bucket=f'{self.infer_Bucket_name}',
                Prefix=f'{inference_dir}', Delimiter=f'/')['CommonPrefixes']))
        except Exception as err:
            s3_list_objects = [{'Prefix': f'{inference_dir}l/'},
                            {'Prefix': f'{inference_dir}r/'}]
            # basic(True) / shor_tall(False) 
            if use_flag:
                log.error(f's3_list_objects) "{inference_dir}" error({str(err)})')
                log.info(f's3_list_objects) 강제 적용 ({s3_list_objects}) success')
            
            s3_flag = False
        return s3_list_objects, s3_flag
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

    ## 왼쪽오른쪽 xml 라인을 가져와서 박스의 가운데가 선을 넘으면 remove 한다.
    def get_front_corr(self, box, line_path, num = 4):
        line_path = line_path + '/line.xml'
        left_right_line = self.get_lines(line_path, top_bot = False)

        c1 = int(left_right_line[0][0])
        c2 = int(left_right_line[1][0])
        final_box = []

        for i in box:
            x_coor = self.center_point(i, xy = 'x')
            if x_coor > c1 and x_coor < c2:
                 final_box.append(i)

        rst = sorted(final_box, key=lambda x : x[0])
        # if lr[-1] == 'l':
        #     rst = sorted(box, key=lambda x : x[0], reverse=False)
        # else:
        #     rst = sorted(box, key=lambda x : x[0], reverse=True)
        # rst = sorted(rst[:num])
        return rst

    # Inference func
    def infer(self, ):
        try:
            start_time = time.time()
            device_shelf_list = self.dao.get_device_shelf(self.companyId, self.storeId, self.deviceId)

            '''
            line 폴더 유무에 따라 short/tall flag 설정 (True: short/tall 로직 수행 / False: 기존 로직 수행)
            '''
            short_tall_dir = f'inference/lines/{self.companyId}/{self.storeId}/{self.deviceId}/'
            st_s3_list_objects, st_s3_flag = self.get_s3_objects(short_tall_dir, False)
            is_short_tall = st_s3_flag

            shelf_dic = {f: {'shelf_model': None, 'cell_count': None} for f in
                         device_shelf_list.shelf_floor.values.tolist()}

            img_dic = {sf: {'main': [], 'empty': []} for sf in device_shelf_list.shelf_model.values.tolist()}

            result_dic = {sf: {'main': [], 'empty': [], 'total_main': [], 'total_empty': [], 'log': []} for sf in
                          device_shelf_list.shelf_model.values.tolist()}

            if is_short_tall:
                short_tall_dic = {sf : {'image': [], 'lines': []} for sf in device_shelf_list.shelf_model.values.tolist()}
                short_tall_box_dirs = []

            # Cells 의 master_pkey Get -> dict
            cells_master_dict = self.dao.get_designs_pkey_master(self.companyId, self.storeId, self.deviceId)

            # Pog List
            pog_list = []
            inference_box_list = []
            for index, device_shelf in device_shelf_list.iterrows():
                floor = device_shelf['shelf_floor']
                shelf_model = device_shelf['shelf_model']
                save_img_path = f'logs/saved_img/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'
                inference_dir = f'inference/boxes/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/'

                if self.save_log_to_s3 == True:
                    s3_list_objects, s3_flag = self.get_s3_objects(inference_dir, True)
                    cameras = list(map(lambda x: os.path.basename(os.path.normpath(x['Prefix'])), s3_list_objects))
                else:
                    # Local insert setting
                    cameras = sorted(os.listdir(f'inference/boxes/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}'))

                floor_crop_empty = []
                floor_crop_main = []

                # camera pictures
                for camera in cameras:
                    try:
                        # start_time = time.time()
                        img = self.re_img.get(f'{self.companyId}_{self.storeId}_{self.deviceId}_f{floor}_cam{camera}')
                        redis_box_key = f'inference/boxes/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}'
                        main_images, em_images, main_boxes= self.img_preprocess(redis_box_key, img, save_img_path, camera)
                        
                        inference_box_list.extend(main_boxes)
                        floor_crop_empty = floor_crop_empty + em_images
                        floor_crop_main = floor_crop_main + main_images

                        if is_short_tall:
                            redis_line_key = f'inference/lines/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}'
                            short_tall_box_dirs.append(f'logs/saved_shortall_box/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{camera}')
                            short_tall_dic[shelf_model]['image'] += [('image', img)]
                            short_tall_dic[shelf_model]['lines'] += [redis_line_key]
                    except Exception as err:
                        log.error(f'cameras images redis get Error({str(err)})')
                        log.error(traceback.format_exc())

                shelf_dic[floor]['shelf_model'] = shelf_model
                shelf_dic[floor]['cell_count'] = len(floor_crop_main)
                img_dic[shelf_model]['main'] += floor_crop_main
                img_dic[shelf_model]['empty'] += floor_crop_empty
            '''
            층별로 생성됨
            img_dic = {
                '00766_w_dri' : {
                    'main' : [img1,img2,img3,...],
                    'empty' : [img1,img2,img3,...] 
                }
            }
            '''
            # 인퍼런스 시작
            for model_name, model_value in img_dic.items():
                start_time = time.time()
                for empty_main_name, empty_main_imgs in model_value.items():
                    predict_img = np.array(empty_main_imgs)

                    channel = grpc.insecure_channel(f'{config_by_name[Config.BOILERPLATE_ENV].EC2_INFERENCE_IP}:{config_by_name[Config.BOILERPLATE_ENV].EC2_INFERENCE_PORT}')
                    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

                    request = predict_pb2.PredictRequest()
                    request.model_spec.name = f'{model_name}_{empty_main_name}'
                    if empty_main_name == 'empty':
                        request.model_spec.name = f'empty_total'
                    request.inputs['input_1'].CopyFrom(
                        make_tensor_proto(predict_img, shape=predict_img.shape, dtype=float))
                    result = stub.Predict(request, 60.0)  # 10 secs timeout

                    outputs = result.outputs
                    detection_classes = outputs["dense"]
                    detection_classes = make_ndarray(detection_classes)
                    score = np.argmax(detection_classes, axis=1)

                    model_label_path = f'inference/model_and_label/{model_name}/{empty_main_name}_labels.txt'
                    if self.save_log_to_s3 == True:
                        model_label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=model_label_path)['Body']
                        main_df = pd.read_csv(model_label_data, sep=' ', index_col=False, header=None)

                    else:
                        main_df = pd.read_csv(model_label_path, sep=' ', index_col=False, header=None)
                    main_class_names = sorted(main_df[0].tolist())
                    predict_product_name = list(map(lambda x: main_class_names[x], score))
                    result_dic[model_name][empty_main_name] = predict_product_name

                    if empty_main_name == 'main':
                        model_classes = main_class_names
                log.info(f'---{time.time() - start_time} - grpc()')
                if is_short_tall: 
                    try:
                        api_info = self.re.get(f'cigar_api_info')
                        if api_info:
                            api_info = eval(api_info)
                    except Exception as err:
                        api_info['SHORT_TALL_DETECTION_URL'] = 'https://125.132.250.227/predict/short_tall'
                        api_info['SHORT_TALL_DETECTION_KEY'] = ''
                        api_info['SHORT_TALL_DETECTION_USERID'] = 'ShortTall'
                        log.warning(f'[short_tall_api_info] Redis Get Error({str(err)}) | 강제 api_info 적용 ({api_info})')

                    response = requests.post(api_info['SHORT_TALL_DETECTION_URL'], files=short_tall_dic[model_name]['image'],
                                             headers={"X-API-KEY": api_info['SHORT_TALL_DETECTION_KEY'],
                                                      "Userid": api_info['SHORT_TALL_DETECTION_USERID']}, verify=False)
                    #response = requests.post(f'http://{config_by_name[Config.BOILERPLATE_ENV].EC2_OBJECT_DETECTION_IP}:{config_by_name[Config.BOILERPLATE_ENV].EC2_OBJECT_DETECTION_PORT}/predict/short_tall', files=short_tall_dic[model_name]['image'])
                    short_tall_tags = []
                    iidx = 0

                    for raw_boxes, line, d in zip(response.json()['res'], short_tall_dic[model_name]['lines'], short_tall_box_dirs):
                        boxes = self.refine_coor(raw_boxes)
                        boxes = self.get_front_corr(boxes, line)
                        ## boxes = [[0, 0, 862, 1080], [780, 219, 897, 910]]
                        ## inference_box_list[iidx*4:iidx+1*4] = [(43, 0, 217, 853), (264, 0, 515, 973), (600, 1, 948, 1079), (1135, 3, 1642, 1080)]
         
                        util.save_xml(d, self.trDate, boxes, self.s3, True, f'{self.save_img_Bucket_name}')
                        short_tall_tags += self.get_short_tall_tag(boxes, line, inference_box_list[iidx*4:(iidx+1)*4])
                        iidx += 1
                        
                # result_dic[model_name]['total_pure'] = list(map(lambda x: x[0] if x[0] == 'empty' else x[1] , zip(result_dic[model_name]['empty'], result_dic[model_name]['main'])))
                # design_dict = self.dao.get_design_table(result_dic[model_name]['total_pure'])
                # result_dic[model_name]['total_db_output']  = [design_dict.get(ir, -1) for ir in result_dic[model_name]['total_pure']]
                design_dict = self.dao.get_design_table(result_dic[model_name]['main'])
                design_device_dict = self.dao.get_device_master_pkey(self.companyId, self.storeId, self.deviceId)
                design_main = []
                design_empty = []
                design_result = []
                design_st_master = []
                design_st_label = []
                design_log = []
                msg_st_list = []    # 숏/톨 강제 pog Log 메시지 리스트 저장.
                kakao_msg_st_list = [] # 숏/톨 강제 pog 카톡 메시지 리스트 저장.
                for idx, (main_label, empty_label) in enumerate(
                        zip(result_dic[model_name]['main'], result_dic[model_name]['empty'])):
                    design_pkey = design_dict.get(main_label, 0) # design_pkey 없으면 0

                    # Pog List append
                    pog_list.append(cells_master_dict[idx]['design_infer_label'])

                    if empty_label == 'empty':  # empty일때 0, product일때 1
                        design_empty.append(0)
                        design_result.append(0)
                        design_st_master.append(0)
                        design_st_label.append(0)
                        design_log.append('empty')
                        msg_st_list.append(0)
                        kakao_msg_st_list.append(0)
                    else:
                        design_empty.append(1)
                        res_design_pkey = design_dict.get(main_label, -1)
                        
                        # 각 Cells의 design_master_pkey 값과 다를 경우 /
                        # 'lc' 모드가 아닐 경우 /
                        # 디바이스의 마스터들의 키값에 포함이 안될 경우
                        # Vision result -> DB Master Key 값으로 대체.
                        if (cells_master_dict[idx]['design_pkey_master'] != design_pkey) and \
                            (cells_master_dict[idx]['inference_mode'] != 'lc') and \
                            (design_pkey not in design_device_dict.keys()):
                            log_msg = f"[Vision 결과가 다름 / cell_pkey({cells_master_dict[idx]['cell_pkey']}) / " +\
                                f"shelf_floor({cells_master_dict[idx]['shelf_floor']})|" +\
                                f"cell_column({cells_master_dict[idx]['cell_column']})] : " +\
                                f"Vision 결과({main_label}, {design_pkey}) / " +\
                                f"DB 결과({cells_master_dict[idx]['design_infer_label']}, " +\
                                f"{cells_master_dict[idx]['design_pkey_master']}) / " +\
                                f"front_pkey({cells_master_dict[idx]['design_pkey_front']})"
                            
                            # 강제 Pog 전환 로그 결과 msg list 처리.
                            if is_short_tall:
                                msg_st_list.append(log_msg)
                            else:
                                log.warning(log_msg)

                            retry_cnt = 0
                            while retry_cnt < 3:
                                try:
                                    vision_dao = DAO()
                                    # 비젼 goods 정보
                                    vs_goods = vision_dao.get_goods_by_design_pkey(design_pkey)
                                    # DB master_pkey goods 정보
                                    db_goods = vision_dao.get_goods_by_design_pkey(int(cells_master_dict[idx]['design_pkey_master']))
                                    # DB first_pkey goods 정보
                                    if str(cells_master_dict[idx]['design_pkey_front']) != 'nan':
                                        f_goods = vision_dao.get_goods_by_design_pkey(int(cells_master_dict[idx]['design_pkey_front']))
                                    else:
                                        f_goods = ('empty', 'empty')

                                    try:
                                        max_floor = vision_dao.get_max_floor(str(cells_master_dict[idx]['device_pkey']))
                                        floors = (max_floor + 1) - int(cells_master_dict[idx]['shelf_floor'])
                                        columns = int(cells_master_dict[idx]['cell_column']) + 1

                                        msg = f"- {floors}층 {columns}칸 비전 확인 요망\n" + \
                                            f"(해당 디바이스에 없는 상품을 비젼으로 인식하여, 해당 셀의 POG로 변경 하였습니다.)\n" + \
                                            f"POG : {db_goods[1]}({db_goods[0]})\n" + \
                                            f"비전 결과 : {vs_goods[1]}({vs_goods[0]})\n" + \
                                            f"이전 거래 비전 결과 : {f_goods[1]}({f_goods[0]})\n\n"
                                        
                                        if is_short_tall:
                                            kakao_msg_st_list.append(msg)
                                        else:
                                            #self.message.send_alimtalk("scope_error", self.trDate, msg, "interminds", self.header)
                                            resp = self.message.send_slack(self.trDate, msg)
                                            log.info(f'[Slack] Send Success')
                                    except Exception as exx:  # 카톡 알림 발송 오류
                                        log.error(f'[kakaotalk / slack] error {exx}')
                                        log.error(traceback.format_exc())
                                except Exception as e:
                                    vision_dao.session.rollback()
                                    vision_dao.session.close()
                                    del vision_dao
                                    retry_cnt += 1
                                    log.error(f'vision Setting Error({str(e)}) / {retry_cnt}회')
                                else:
                                    # Design_Pkey
                                    res_design_pkey = int(cells_master_dict[idx]['design_pkey_master'])
                                  
                                    # short/tall 로직이 아닐 경우 적용.
                                    if not is_short_tall:
                                        # Design_Label
                                        main_label = cells_master_dict[idx]['design_infer_label']
                                        # Main Desig_Pkey
                                        design_pkey = int(cells_master_dict[idx]['design_pkey_master'])
                                    # short/tall 로직 (강제 key 전환)
                                    else:
                                        design_st_master.append(int(cells_master_dict[idx]['design_pkey_master']))
                                        design_st_label.append(cells_master_dict[idx]['design_infer_label'])
                                    # DB session Close
                                    vision_dao.session.close()
                                    del vision_dao
                                    break
                        else:
                            kakao_msg_st_list.append(0) # 강제 Pog 전환 없을 경우 (0) / 숏톨 카톡 메시지
                            msg_st_list.append(0)       # 강제 Pog 전환 없을 경우 (0) / 숏톨 로그 메시지
                            design_st_master.append(0)  # vision 오탐 강제 판단을 위한 선별 값(0)
                            design_st_label.append(0)   # vision 오탐 강제 판단을 위한 선별 값(0)

                        design_result.append(res_design_pkey)   # redis (inf_result) set
                        design_log.append(main_label)

                    # total_main insert.
                    design_main.append(design_pkey)        
                                    
                result_dic[model_name]['total_main'] = design_main
                result_dic[model_name]['total_empty'] = design_empty
                result_dic[model_name]['log'] = design_log

                if is_short_tall:
                    short_tall_tags = self.fill_empty_part(short_tall_tags, result_dic[model_name]['empty'])
                    result_dic[model_name]['total_main'], short_tall_labels = self.get_short_tall_pkeys_and_labels(
                        self.dao.get_design_and_tag(model_classes),
                        result_dic[model_name]['main'],
                        *short_tall_tags
                        )

                    # short/tall vision 오탐 강제 pog 변경 (master_pkey)
                    try:
                        for idx, (total_main, master_pkey, master_label, log_msg, kakao_msg) in enumerate(zip(result_dic[model_name]['total_main'], design_st_master, design_st_label, msg_st_list, kakao_msg_st_list)):
                            if (master_pkey == 0) or (total_main == master_pkey) or (total_main in (design_device_dict.keys())):
                                continue

                            log.warning(log_msg)
                            # Kakao / Slack Msg Send
                            try:
                                #self.message.send_alimtalk("scope_error", self.trDate, kakao_msg, "interminds", self.header)
                                resp = self.message.send_slack(self.trDate, kakao_msg)
                                log.info(f'[Slack] Send Success')
                            except Exception as exx:  # 카톡 알림 발송 오류
                                log.error(f'[kakaotalk / slack] error {exx}')

                            # 숏/톨 (Line) 로직 후 강제 Pog 전환.
                            result_dic[model_name]['total_main'][idx] = master_pkey
                            short_tall_labels[idx] = master_label
                    except Exception as err:
                        log.error(f'[Short/Tall] 강제 Pog 전환 로직 Error({str(err)})')

            # log_str = f"[CV]c:{self.companyId}/s:{self.storeId}/d:{self.deviceId}".center(100)
            self.log_str = "\n(Master Pog List)"
            short_tall_tag_log = "\n(Short Tall Tag Result)"
            short_tall_label_log = "\n(Short Tall Label Result)"
            temp_log_str = "\n\n(Label Result)"
            total_log_str = "\n\n(Total Result)"
            
            # total result log를 위한 층별 master pog 리스트 생성
            try:
                total_dic = dict()
                for pog in cells_master_dict.values():
                    if pog['shelf_floor'] in total_dic.keys():
                        total_dic[pog['shelf_floor']].append(pog)
                    else:
                        total_dic[pog['shelf_floor']] = list()
                        total_dic[pog['shelf_floor']].append(pog)
            except Exception as e:
                log.error(f"Created master pog list for total result log message error [{e}]")

            # Redis Set (inf) 고유 키
            total_floor = len(shelf_dic)
            total_short_tall = dict()
            for sh_floor, sh_value in shelf_dic.items():
                sh_model = sh_value['shelf_model']
                sh_count = sh_value['cell_count']
                now_floor = f'[{str(total_floor - sh_floor)}층]' # 층 확인
                self.re.set('{}_{}_{}_f{}_inf_main'.format(self.companyId, self.storeId, self.deviceId, sh_floor),
                       json.dumps(result_dic[sh_model]['total_main'][:sh_count]))
                self.re.set('{}_{}_{}_f{}_inf_empty'.format(self.companyId, self.storeId, self.deviceId, sh_floor),
                       json.dumps(result_dic[sh_model]['total_empty'][:sh_count]))
                self.re.set('{}_{}_{}_f{}_inf_result'.format(self.companyId, self.storeId, self.deviceId, sh_floor),
                       json.dumps(design_result[:sh_count]))
                # re.set('{}_{}_{}_f{}_inf_result'.format(self.companyId, self.storeId, self.deviceId, sh_floor), json.dumps(result_dic[sh_model]['total_db_output'][:sh_count]))
                # log.info(f'[CV]c:{self.companyId}/s:{self.storeId}/d:{self.deviceId}:   {result_dic[sh_model]["total_pure"][:sh_count]} ')
                # log_str += '\n' + str(result_dic[sh_model]["total_pure"][:sh_count])
                self.log_str += '\n' + str(now_floor) + str(pog_list[:sh_count])
                temp_log_str += '\n' + str(now_floor) + str(result_dic[sh_model]['log'][:sh_count])

                if is_short_tall:
                    total_short_tall[sh_floor] = short_tall_labels[:sh_count] # total_result log 메세지를 위한 short tall 층별 결과
                    short_tall_tag_log += '\n' + str(now_floor) + str(short_tall_tags[:sh_count])
                    short_tall_label_log += '\n' + str(now_floor) + str(short_tall_labels[:sh_count])
                    short_tall_tags = short_tall_tags[sh_count:]
                    short_tall_labels = short_tall_labels[sh_count:]
                    
                
                # Total Result Log Logic 추가.
                # mix : mix 결과
                # LC :  empty -> 비젼결과 empty
                #       not empty -> master pog
                # short tall이 존재하는 경우는 short tall label 결과, 존재하지않으면 label result를 기준으로 진행
                total_temp_list = list()
                try:
                    for idx in range(len(total_dic[sh_floor])):
                        temp = total_dic[sh_floor][idx]
                        temp_total_result = total_short_tall[sh_floor][idx] if is_short_tall else result_dic[sh_model]['log'][idx]
                        if temp['inference_mode'] == 'mix':
                            total_temp_list.append(temp_total_result)
                        else:
                            if temp_total_result != 'empty':
                                total_temp_list.append(pog_list[idx])
                            else:
                                total_temp_list.append(temp_total_result)
                                
                    total_log_str += f"\n{now_floor}{total_temp_list}"
                    
                except Exception as e:
                    total_log_str = f"\n"
                    log.error(f"Created total result log message error [{e}]")

                result_dic[sh_model]['total_main'] = result_dic[sh_model]['total_main'][sh_count:]
                result_dic[sh_model]['total_empty'] = result_dic[sh_model]['total_empty'][sh_count:]
                result_dic[sh_model]['log'] = result_dic[sh_model]['log'][sh_count:]
                pog_list =  pog_list[sh_count:]
                design_result = design_result[sh_count:]

            self.log_str += temp_log_str

            if is_short_tall:
                self.log_str += '\n' + short_tall_tag_log
                self.log_str += '\n' + short_tall_label_log
                
            self.log_str += total_log_str # add total result log msg
                
        except Exception as ex:
            err_code = traceback.format_exc()
            log.error(f"[inference error] {err_code}")
            if self.conf_email_alarm == True:
                try:
                    log.info("send inference error code to kakaotalk")
                    context = f"- 인퍼런스 서버 오류입니다.\n" + \
                              f"- instance id : {ec2_metadata.instance_id}\n" + \
                              f"- private ip : {ec2_metadata.private_ipv4}\n" + \
                              f"- public ip : {ec2_metadata.public_ipv4}\n" + \
                              f"- region : {ec2_metadata.region}\n\n" + \
                              f"{err_code}"
                except Exception as e:  # ec2_metadata를 가져올 수 없음
                    context = f"- 인퍼런스 서버 오류입니다.\n" + \
                              f"{err_code}"
                finally:
                    log.info(f"[kakaotalk] send messge \n {context}")
                    try:
                        self.message.send_alimtalk("server_error", self.trDate, context, "interminds")
                    except Exception as exx:  # 카톡 알림 발송 오류
                        log.error(f'[kakaotalk] error {exx}')
                        log.error(traceback.format_exc())

            # abort
            self.res['abort'] = {'code': 602, 'msg': f'inference Error({str(ex)})'}  # 602 abort Error

    # 이미지 프로세서 func
    def img_preprocess(self, redis_box_key, img, save_img_path, camera):
        # Folder Path 정의
        main_crop_path = f'{os.path.split(save_img_path)[0]}/main/'
        empty_crop_path = f'{os.path.split(save_img_path)[0]}/empty/'
        save_full_path = f'{save_img_path}/{camera}/{self.trDate.split("_")[0]}'  # 이미지 저장소 파일(날짜별)

        # Folder 생성
        util.createFolder(main_crop_path)
        util.createFolder(empty_crop_path)
        util.createFolder(save_full_path)

        encoded_img = np.frombuffer(img, dtype=np.uint8)
        image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.save_img_s3_or_local(f'{save_full_path}/{self.trDate}.jpg', image)
        _, frame = cv2.imencode('.jpg', image)
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        empty_boxes = self.get_boxes(redis_box_key + '/empty.xml')
        main_boxes = self.get_boxes(redis_box_key + '/main.xml')
        
        for index, boxes in enumerate([main_boxes, empty_boxes]):
            each_column = list(map(lambda j: image[j[1]:j[3], j[0]:j[2]], boxes))
            try:
                # main
                if index == 0:
                    main_images = list(map(lambda j: cv2.resize(j, self.resize), each_column))
                    # for idx, images in enumerate(main_images):
                    #     # cv2.imwrite(f'{main_crop_path}/{self.trDate}_{floor}_{camera}_{idx}.jpg', images)
                    #     # cv2.imwrite(f'{main_crop_path}/{floor}_{camera}_{idx}.jpg', images)
                    #     self.save_img_s3_or_local(f'{main_crop_path}/{floor}_{camera}_{idx}.jpg', image, save_log_to_s3=False)
                    main_images = list(map(lambda j: preprocess_input(j), main_images))
                # empty
                else:
                    empty_images = list(map(lambda j: cv2.resize(j, self.resize), each_column))
                    # for idx, images in enumerate(empty_images):
                    #     # cv2.imwrite(f'{empty_crop_path}/{self.trDate}_{floor}_{camera}_{idx}.jpg', images)
                    #     # cv2.imwrite(f'{empty_crop_path}/{floor}_{camera}_{idx}.jpg', images)
                    #     self.save_img_s3_or_local(f'{empty_crop_path}/{floor}_{camera}_{idx}.jpg', image, save_log_to_s3=False)
                    empty_images = list(map(lambda j: preprocess_input(j), empty_images))

            except Exception as e:
                print(e)
                log.error(traceback.format_exc())

        return main_images, empty_images, main_boxes

    # 디렉토리 생성 func
    # def createFolder(self, directory):
    #     try:
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #     except OSError:
    #         traceback.print_exc()

    # 이미지 파일 (s3) 또는 (local) 에 저장 func
    def save_img_s3_or_local(self, save_path, image):
        if self.save_log_to_s3 == True:
            self.s3.put_object(Body=cv2.imencode('.jpg', image)[1].tostring(), Bucket=self.save_img_Bucket_name, Key=save_path)
        else:
            cv2.imwrite(f'{save_path}', image)

    # Close Inference Xml(main, empty) S3 Save.
    def save_infer_xml_s3(self, label_path, xml_result):
        try:
            temp_dir = str(label_path).split('/')
            floor = temp_dir[5]
            cam = temp_dir[6]
            file_name = str(temp_dir[-1:][0])
            saved_path = f'logs/inference_box_xml/{self.companyId}/{self.storeId}/{self.deviceId}/{floor}/{cam}/{self.str_ymd}'

            # Payment Log 내용 작업.
            if floor not in self.save_xml_path_dict["all_floor_save_xml_path"].keys():
                self.save_xml_path_dict["all_floor_save_xml_path"][floor] = [saved_path + '/' + self.str_hms + '_' + file_name]
            else:
                self.save_xml_path_dict["all_floor_save_xml_path"][floor].append(saved_path + '/' + self.str_hms + '_' + file_name)

            # xml 저장.
            util.save_xml(saved_path, self.str_hms + '_' + file_name.split('.')[0], xml_result, self.s3, True, self.save_img_Bucket_name)
        except Exception as err:
            log.error(f'save infer xml error{err}')

    # Inference Box xml 적용 func
    def get_boxes(self, label_path):
        if self.save_log_to_s3 == True:
            try:
                label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=label_path)['Body']
                root_1 = minidom.parse(label_data)
            except Exception as err:
                log.error(f'[get_boxes] label_path({label_path}) No Such Error)')
                try:
                    # 초기 xml 사용
                    xml_name = str(label_path).split('/')[-1:][0]  # empty.xml / main.xml
                    label_path = f"inference/boxes/init_box/{xml_name}"
                    label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=label_path)['Body']
                    root_1 = minidom.parse(label_data)
                except Exception as err:
                    log.error(f'[get_boxes] 초기화 label({label_path}) 적용 Error({str(err)})')
                else:
                    log.info(f'[get_boxes] 초기화 label({label_path}) 적용 success')
        else:
            xml_path = os.path.join(label_path)
            root_1 = minidom.parse(xml_path)
        bnd_1 = root_1.getElementsByTagName('bndbox')
        result = []
        xml_result = []
        for i in range(len(bnd_1)):
            xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
            ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
            xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
            ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
            result.append((xmin, ymin, xmax, ymax))
            xml_result.append([xmin, ymin, xmax, ymax])
        
        try:
            # infer xml(main, empty) S3 save / Threading Proc
            t1 = threading.Thread(target=self.save_infer_xml_s3, args=(label_path, xml_result,))
            t1.daemon = True 
            t1.start()
        except Exception as err:
            log.error(f'[save_infer_xml_s3] Threading process Error ({str(err)})')

        return result

    def get_lines(self, label_path, top_bot = True):
        '''
        -> [top_line (xmin, ymin, xmax, ymax), btm_line (xmin, yumin, xmax, ymax)]
        '''
        if self.save_log_to_s3 == True:
            try:
                label_data = self.s3.get_object(Bucket= f'{self.infer_Bucket_name}', Key= label_path)['Body']
                root_1 = minidom.parse(label_data)
            except Exception as err:
                log.error(f'[get_boxes] label_path({label_path}) No Such Error)')
                try:
                    # 초기 xml 사용
                    label_path = f"inference/lines/line_sample.xml"
                    label_data = self.s3.get_object(Bucket=f'{self.infer_Bucket_name}', Key=label_path)['Body']
                    root_1 = minidom.parse(label_data)
                except Exception as err:
                    log.error(f'[get_lines] 초기화 label({label_path}) 적용 Error({str(err)})')
                else:
                    log.info(f'[get_lines] 초기화 label({label_path}) 적용 success')
        else:
            xml_path = os.path.join(label_path)
            root_1 = minidom.parse(xml_path)
        if top_bot:
            top = root_1.getElementsByTagName('top')
            btm = root_1.getElementsByTagName('bottom')
            result = []
            for node in (top, btm):
                xmin = int(node[0].childNodes[1].childNodes[0].nodeValue)
                ymin = int(node[0].childNodes[3].childNodes[0].nodeValue)
                xmax = int(node[0].childNodes[5].childNodes[0].nodeValue)
                ymax = int(node[0].childNodes[7].childNodes[0].nodeValue)
                result.append((xmin,ymin,xmax,ymax))
        else:
            left = root_1.getElementsByTagName('left')
            right = root_1.getElementsByTagName('right')
            result = []
            for node in (left, right):
                xmin = int(node[0].childNodes[1].childNodes[0].nodeValue)
                ymin = int(node[0].childNodes[3].childNodes[0].nodeValue)
                xmax = int(node[0].childNodes[5].childNodes[0].nodeValue)
                ymax = int(node[0].childNodes[7].childNodes[0].nodeValue)
                result.append((xmin,ymin,xmax,ymax))
        return result

# api 요청 (Door Open)
def door_open_req(companyId, storeId, deviceId):
    try:
        request_url = f'http://localhost:5000/admin_door_opened'

        res = requests.post(request_url,
                            json={'companyId': companyId, 'storeId': storeId, 'deviceId': deviceId},
                            verify=False, timeout=100)
        return res.json()
    except Exception as err:
        log.error(f'[route_{devkind}] door_open 요청 Error ({str(err)})')
        log.error(traceback.format_exc())

# api 요청 (tf_model)
@fridge_bp.route('/tf_model', methods=['POST'])
def tf_model():
    log.info(f'tf_model start')
    result = {}
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

        if work_user == 'manager':
            header = "(관리자 모드)"
        elif work_user == 'interminds':
            header = "(테스트 모드)"
        else:
            header = "(사용자 모드)"

        trNo = request.json['trNo']

        start_time = time.time()

        # Db class
        dao = DAO()

        # inference logic 성공 여부 (default : True)
        infer_work_flag = True

        # Error code 초기 값(200 정상)
        error_code = 200
        
        # door close class
        
        door_closed = FridgeDoorClosed(companyId, storeId, deviceId, work_user,
                                       trDate, trade_date, trade_time, trNo, trResponseDate)
        log.info('door_closed 객체생성')
        # infer class
        f_inf = FridgeInference(companyId, storeId, deviceId, trDate, door_closed.re, header, work_user)
        log.info('FridgeInference 객체생성')

        # manager
        if work_user == 'manager':
            try:
                # infer func
                start_time = time.time()
                f_inf.infer()
                log.info(f'---{time.time() - start_time} - f_inf.infer()')
                log_str = f_inf.log_str
                log_str += '\n' + f"model inf: {time.time() - start_time}".center(100)
                util.LogGetter.log += log_str

                # infer abort 처리
                if f_inf.res['abort']['code'] != 200:
                    error_code = f_inf.res['abort']['code']
                    log.error(f'Inference Server Error ({error_code})')
                    infer_work_flag = False
                    #raise Exception(f_inf.res['abort']['msg'])

                # save xml path 공유.
                start_time = time.time()
                door_closed.set_save_xml_path(f_inf.save_xml_path_dict)
                log.info(f'---{time.time() - start_time} - set_save_xml_path')

                # product_count() / infer 동작 여부 (True / False)
                start_time = time.time()
                result = door_closed.product_count(infer_work_flag)
                log.info(f'---{time.time() - start_time} - product_count')

                # product count Error (Retry Open -> Close)
                if result['abort']['code'] != 200:
                    dao.session.rollback()
                    util.LogGetter.log += '\n' + f"[admin_door_closed] 관리자 문 닫기 무게 오류, " \
                                            f"open 후 한번 더 closed 실행".center(100, ' ') + '\n'
                    
                    # 로직 재수행, 인퍼런스 동작 여부 초기화
                    infer_work_flag = True

                    # 관리자로 문닫고 에러나면 문 열고 한번 더 돌리는 로직
                    # infer func
                    start_time = time.time()
                    f_inf.infer()
                    log.info(f'---{time.time() - start_time} - f_inf.infer()')
                    log_str = f_inf.log_str
                    log_str += '\n' + f"model inf: {time.time() - start_time}".center(100)
                    util.LogGetter.log += log_str

                    # infer abort 처리
                    if f_inf.res['abort']['code'] != 200:
                        error_code = f_inf.res['abort']['code']
                        log.error(f'Inference Server Error ({error_code})')
                        infer_work_flag = False
                        #raise Exception(f_inf.res['abort']['msg'])
                    start_time = time.time()
                    res = door_open_req(companyId, storeId, deviceId)
                    log.info(f'---{time.time() - start_time} - door_open_req')

                    if 'SUCCESS' == res['resultMsg']:
                        # save xml path 공유.
                        start_time = time.time()
                        door_closed.set_save_xml_path(f_inf.save_xml_path_dict)
                        log.info(f'---{time.time() - start_time} - set_save_xml_path')
                        # product_count() / infer 동작 여부 (True / False)
                        
                        start_time = time.time()
                        result = door_closed.product_count(infer_work_flag)
                        log.info(f'---{time.time() - start_time} - product_count')
                    else:
                        log.error(f"[route_{devkind}] door_open request Error ({str(res['resultMsg'])})")
            except Exception as e:
                result['abort'] = {'code': f'{error_code}', 'msg': f'{str(e)}'}
                log.error(f'[tf_model] Error {str(e)}')
                log.error(traceback.format_exc())
        # customer
        else:
            # infer func
            start_time = time.time()
            f_inf.infer()
            log.info(f'---{time.time() - start_time} - f_inf.infer()')
            log_str = f_inf.log_str
            log_str += '\n' + f"model inf: {time.time() - start_time}".center(100)
            util.LogGetter.log += log_str

            # infer abort 처리
            if f_inf.res['abort']['code'] != 200:
                error_code = f_inf.res['abort']['code']
                log.error(f'Inference Server Error ({error_code})')
                infer_work_flag = False
                #raise Exception(f_inf.res['abort']['msg'])

            # save xml path 공유.
            
            start_time = time.time()
            door_closed.set_save_xml_path(f_inf.save_xml_path_dict)
            log.info(f'---{time.time() - start_time} - set_save_xml_path')

            # product_count() / infer 동작 여부 (True / False)
            start_time = time.time()
            result = door_closed.product_count(infer_work_flag)
            log.info(f'---{time.time() - start_time} - product_count')
        # Total Log Write
        log.info(util.LogGetter.log)
        dao.session.commit()

        # Run (0) Redis Set --TODO 임시로 생성 차후 로직 수정 후 삭제 예정
        try:
            redis_key = f'{companyId}_{storeId}_{deviceId}_run'
            door_closed.re.set(redis_key, 0)
        except Exception as err:
            log.error(traceback.format_exc())
            log.error(f'Redis Key ({redis_key}) Error')

    except Exception as e:
        # abort 처리.
        result['abort'] = {'code': f'{error_code}', 'msg': f'{str(e)}'}
    finally:
        del door_closed
        del f_inf
        dao.session.close()
    return json.dumps(result, ensure_ascii=False)

@fridge_bp.route('/')
@fridge_bp.route('/index')
def index():
    return 'hello'

# def align_center(log_str, max_len=100):
#     log_str = log_str.split('\n')
#     return '\n'.join([row.center(max_len, ' ') for row in log_str])

