# -*- coding:utf-8 -*-
import os
import redis
import json
import ssl
import time
import requests
import logging.config
import logging

#import models
from . import models
import traceback
import ast
import urllib3
from datetime import datetime
import boto3

from keys import keys

from .data_access import DAO
#set_models(models)
from .lc_inf import OrderList
from .loadcell_snapshot import snapshot
from flask import Flask, request, render_template, redirect, Blueprint
# from .log_adapter import StyleAdapter
import app.log_adapter
from flask_cors import CORS
from sqlalchemy.exc import InvalidRequestError
from distutils.util import strtobool
from .log_designate import LogDesignate
# abort class
from .abort_err_code import *

from .common import get_status_info

from .config import Config
from .config import config_by_name

# from .log_getter import LogGetter
from .alarm import Alarm
import app.util as util
from .client_socket import Client_Socket

# str_log_file_name = 'status.log'
# logging.config.fileConfig('logging.conf', disable_existing_loggers=False, defaults={"str_log_file_name": str_log_file_name})
# logger = StyleAdapter(logging.getLogger("log03"))
root_logger = logging.getLogger()
logger = logging.getLogger('basic_log')
logger_root = logging.getLogger('basic_root_log')
logger_simple = logging.getLogger('simple_log')

logger.info(f'{os.getpid()}|server_main| ########## S T A R T ##########')

re = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
    db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
    password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD, charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
    decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)

bp = Blueprint('main', __name__, url_prefix='/')

# Abort Error Code define.
"""
600 : LoadCell Error
601 : Product_count Error
602 : Inference Error
"""
@bp.errorhandler(600)
def uflr(e):
    return e, 600
@bp.errorhandler(601)
def uflr(e):
    return e, 601
@bp.errorhandler(602)
def uflr(e):
    return e, 602

def main_function():
    print ("MAIN FUNCTION START")

# log 권한 get func
def get_dev_stg_type(storeId, deviceId):

    dao = DAO()
    try:
        # deviceId를 통한 device_storage_type Get ('CD' : 냉장고, 'CG' : 담배, 'AC' : 주류, 'VC' : 백신)
        try:
            device_storage_type = dao.get_device_storage_type(storeId, deviceId)
        except Exception as e:
            logger.warning("[device_storage_type] 'NoneType' -> 'CD' ")
            device_storage_type = 'CD'
        # 담배
        if device_storage_type == 'CG':
            dev_kind = 'cigar'
        # 주류
        # elif device_storage_type == 'AC':
        #     dev_kind = 'alcohol'
        # 백신
        elif device_storage_type == 'VC':
            dev_kind = 'vaccine'
        # 냉장고
        else:
            dev_kind = 'fridge'

        # dev_kind에 따라 (fridge, cigar, alcohol) 로그 분리.
        log = LogDesignate(dev_kind)

        return log, dev_kind
    except Exception as e:
        logger.error(f'[get_device_stg_type] Error {str(e)}')
        logger.error(traceback.format_exc())
        return logger, 'fridge'
    finally:
        dao.session.close()

@bp.route('/check_status', methods=['POST'])
def check_status():
    '''
        이거는 신세계에서 날라옴
        사용자가 인증할 때 날라옴
        냉장고 상태(로드셀, 카메라) 점검하는거
        두가지 종류:
            사용자가 사용할 때     : 스토어, 디바이스 날라옴 : 해당 디바이스만 상태 조회(카메라, 로드셀 멀쩡한지)
            신세계가 그냥 점검할 때 : 스토어 아이디만 날라옴 : 모든 냉장고 상태 조회(카메라, 로드셀 멀쩡한지)
    '''
    '''
        파이가 5분에 한 번씩 레디스에 상태 저장함(층 별 카메라, 로드셀 정상 여부)
        {'0': {'cam': 0, 'lc': 0}, '1': {'cam': 0, 'lc': 0}, '2': {'cam': 0, 'lc': 0}, '3': {'cam': 0, 'lc': 0}}
        레디스에 저장된 값 읽어서 상태 정보 리턴해줌
    '''
    start_time = time.time()
    try:
        dao = DAO()
        # log 권한 요청 check_status 의 경우 basic_log 사용.
        log = LogDesignate('check_status')
        
        storeId = request.json['storeId']
        try:
            deviceId = request.json['deviceId']
        except KeyError:
            deviceId = None
        try:
            companyId = request.json['companyId']
        except KeyError:
            companyId = dao.get_company_id(storeId, deviceId)

        # res = get_store_info()
        deviceId_list = []

        if deviceId != None:
            deviceId_list.append(deviceId)

        if not deviceId_list:
            log.info("paramiter deviceId is null")
            deviceId_list = dao.get_device_operation(companyId, storeId)['device_id'].tolist()

        # logger_simple.info(f'status check deviceId_list : {deviceId_list}'.center(100))
        log_str = f'status check deviceId_list : {deviceId_list}'
        stable_deviceId_list = []
        error_deviceId_list = []

        while deviceId_list:
            status_check_flag = 0
            cell_alert_flag = 0

            deviceId = deviceId_list.pop()
            # check device status
            try:
                byte_status_check = re.get(f'{companyId}_{storeId}_{deviceId}_status_check')
                byte_status_check = ast.literal_eval(byte_status_check)
            except Exception as e:
                log.error(f'Redis Get ({companyId}_{storeId}_{deviceId}_status_check) Error] {e}')
                error_deviceId_list.append(deviceId)
                break

            # log_str = '=' * 17 + str(deviceId) + ' status_check' + '=' *17 + '|' + '=' * 16 + str(deviceId) + ' cell_alert' + '=' * 16
            # logger.info(f'{deviceId} status_check: {byte_status_check}')
            for floor, floor_value in byte_status_check.items():
                for cell, cell_value in floor_value.items():
                    if isinstance(cell_value,dict):
                        for cam_value in cell_value.values():
                            if int(cam_value) == 1:
                                status_check_flag = 1
                    else:
                        if int(cell_value) == 1:
                            status_check_flag = 1

            # check cell alert
            try:
                byte_cell_alert = re.get(f'{companyId}_{storeId}_{deviceId}_cell_alert')
                byte_cell_alert = ast.literal_eval(byte_cell_alert)
            except Exception as e:
                log.error(f'Redis Get ({companyId}_{storeId}_{deviceId}_cell_alert) Error] {e}')
                error_deviceId_list.append(deviceId)
                break

            log_str += '\n'+f'------ {str(deviceId)} status_check ------ '
            
            for key, value in byte_status_check.items():
                # log_str += '\n' + str(key) + 'f : ' + str(value) + ' | ' + str(byte_cell_alert[key])
                log_str += '\n'+f'{str(key)}f : {str(value)}'

            log_str += '\n'+f'------ {str(deviceId)}   cell_alert ------ '
            for floor, floor_value in byte_cell_alert.items():
                log_str += '\n'+f'{str(floor)}f : {str(floor_value)}'
                for cell, cell_value in floor_value.items():
                    if int(cell_value) == 1:
                        cell_alert_flag = 1


            if status_check_flag == 0 and cell_alert_flag == 0:
                stable_deviceId_list.append(deviceId)
            else:
                error_deviceId_list.append(deviceId)
        log.info(util.align_center(log_str))
        log.info(f"check response time : {time.time() - start_time}".center(100))

        if len(error_deviceId_list) == 0:
            response = {
                'resultCode': "000",
                'resultMsg': "SUCCESS",
                "data": {'companyId': companyId, 'storeId': storeId, 'deviceId': stable_deviceId_list}
            }
            log.info(response)
        else:
            response = {
                'resultCode': "001",
                'resultMsg': "FAIL, something wrong on device",
                "data": {'companyId': companyId, 'storeId': storeId, 'deviceId': error_deviceId_list}
            }
            log.error(response)

        return json.dumps(response)
    except TypeError as e:
        log.error(traceback.format_exc())
        abort(400,'{"message":"type error / wrong deviceId or storeId or companyId"}')
    except redis.RedisError as rer:
        log.error(traceback.format_exc())
        abort(400,'{"message":"redis error / wrong deviceId or storeId or companyId"}')
    except Exception as err:
        log.error(traceback.format_exc())
        abort(500, str(err))
    finally:
        dao.session.close()


@bp.route('/door_opened', methods=['POST'])
def door_opened(work_user = 'customer'):
    start = time.time()
    try:
        dao = DAO()
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        try:
            companyId = request.json['companyId']
        except KeyError:
            companyId = dao.get_company_id(storeId, deviceId)

        # log 권한 요청 (냉장고, 담배, 주류, 백신)
        log, dev_kind = get_dev_stg_type(storeId, deviceId)
        
        ##백신냉장고 문열림 알람 추가
        if dev_kind == 'vaccine':
            aa = Alarm(companyId, storeId, deviceId, work_user)
            now = datetime.now()
            date_time = now.strftime('%Y-%m-%d_%H:%M:%S')
            tmpt = aa.send_slack(date_time,":smile:문열림:smile:",dev_kind)
        # 고객 open시 redis run(1) set
        if work_user == 'customer':
            try:
                redis_key = f'{companyId}_{storeId}_{deviceId}_run'
                re.set(redis_key, 1)
            except Exception as err:
                log.error(traceback.format_exc())
                log.error(f'Redis Key ({redis_key}) Error')
        # 매니저 open시 redis run(2) set
        else:
            try:
                run_check = re.get(f'{companyId}_{storeId}_{deviceId}_run')
                run_check = eval(run_check)
            except Exception as err:
                log.error(f'Redis Get Error({err}) -> run_check(0) setting')
                run_check = 0

            if run_check != 1:
                try:
                    redis_key = f'{companyId}_{storeId}_{deviceId}_run'
                    re.set(redis_key, 2)
                except Exception as err:
                    log.error(traceback.format_exc())
                    log.error(f'Redis Key ({redis_key}) Error')

        # 냉장고 로직일 경우만 total snapshot 수행.
        if dev_kind == 'fridge':
            total_snapshot(dao, companyId, storeId, deviceId, work_user)
    
    except redis.RedisError as rer:
        log.error(traceback.format_exc())
        abort(400, f'redis error / wrong set {companyId}_{storeId}_{deviceId}_run')
    except Exception as err:
        log.error(traceback.format_exc())
        abort(500, str(err))
    finally:
        dao.session.close()

    response = {
        'resultCode': "000",
        'resultMsg': "SUCCESS",
        "data": {}
    }
    log.info('[도어 오픈 응답] {:0.3}초 {}'.format(time.time()-start,str(response)))
    return json.dumps(response)

# pass_sales_info
@bp.route('/pass_sales_info', methods=['POST'])
def pass_sales_info():
    start = time.time()

    try:
        companyId = request.json['companyId']
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        destinationIp = request.json['destinationIp']
        destinationPort = request.json['destinationPort']
        data = request.json['data']

        # log 권한 요청 (냉장고, 담배, 주류, 백신)
        log, dev_kind = get_dev_stg_type(storeId, deviceId)

        log.info(f'[{dev_kind}] : {request.data}')

        # # Client Socket Class 생성.
        # try:
        #     sock = Client_Socket('192.168.0.163', 1005)
        # except Exception as e:
        #     logger.error(f'Socket Create Error')
        # else:
        #     # Socket 연결.
        #     c_flag, logmsg = sock.Connect()
        #
        #     if c_flag:
        #         logger.info(logmsg)
        #
        #         senddata = {
        #                         'companyId':f'{companyId}',
        #                         'storeId':f'{storeId}',
        #                         'deviceId':f'{deviceId}',
        #                         'destinationIp':f'{destinationIp}',
        #                         'destinationPort':f'{destinationPort}',
        #                         'data':f'{data}'
        #                     }
        #
        #         # Json Msg Send
        #         send_data = json.dumps(senddata)
        #         logger.info(f'sock send data) {send_data}')
        #
        #         s_flag, logmsg = sock.sendMsg(send_data)
        #         if s_flag:
        #             logger.info(logmsg)
        #
        #             r_flag, recv_msg = sock.recvMsg()
        #             if r_flag == False:
        #                 logger.error(f'recv msg) {recv_msg}')
        #                 # "003" Recv Fail Error
        #                 response = {
        #                     "resultCode": "003",
        #                     "resultMsg": "RECV FAIL"
        #                 }
        #             else:
        #                 logger.info(f'recv msg) {recv_msg}')
        #                 # "000" Success
        #                 response = {
        #                     "resultCode": "000",
        #                     "resultMsg": "SUCCESS"
        #                 }
        #         else:
        #             logger.error(logmsg)
        #             # "002" Send Fail Error
        #             response = {
        #                 "resultCode": "002",
        #                 "resultMsg": "SEND FAIL"
        #             }
        #
        #         # 소켓 연결 해제.
        #         logmsg = sock.DisConnect()
        #         logger.info(logmsg)
        #     else:
        #         logger.error(logmsg)
        #
        #         # "001" Socket Connection Error
        #         response = {
        #                 "resultCode": "001",
        #                 "resultMsg": "SOCKET CONN ERROR"
        #         }

        # TODO Test
        response = {
            "resultCode": "000",
            "resultMsg": "SUCCESS"
        }

        log.info('[{} 매출 정보 수신] {:0.3}초 {}'.format(dev_kind, time.time() - start, str(response)))
        return json.dumps(response)

    except Exception as err:
        log.error(traceback.format_exc())
        abort(500, str(err))

''' 가결제시 테스트용으로 잠시 필요해서 만든부분 '''
@bp.route('/test_opened', methods=['POST'])
def test_opened():
    response = {
        'resultCode': "000",
        'resultMsg': "SUCCESS",
        "data": {}
    }
    return json.dumps(response)

@bp.route('/test_closed', methods=['POST'])
def test_closed():
    data = {"orderList":[{'goodsId': '8809350888168', 'goodsCnt': '1','goodsName':'GET콜드브루아메리카노','goodsPrice':'2000' , 'RowNo': '1', 'ColNo': '1'}, {'goodsId': '8801056192013', 'goodsCnt': '2', 'goodsName':'롯데)칠성사이다P500ml','goodsPrice':'2100','RowNo': '2', 'ColNo': '2'}, {'goodsId': '8806002007298', 'goodsCnt': '3', 'goodsName':'광동)비타500병180ml','goodsPrice':'1300', 'RowNo': '3', 'ColNo': '3'}], "trDate": "20210101100000", "trNo": "1001" }

    response = {
    'resultCode': "000",
    'resultMsg': "SUCCESS",
    'data': data}
    return json.dumps(response, ensure_ascii=False)

@bp.route('/test_closed_1', methods=['POST'])
def test_closed_1():
    data = {"orderList":[{'goodsId': '8809350888168', 'goodsCnt': '1', 'goodsName':'GET콜드브루아메리카노','goodsPrice':'2000', 'RowNo': '1', 'ColNo': '1'}, {'goodsId': '8801056192013', 'goodsCnt': '2', 'goodsName':'롯데)칠성사이다P500ml','goodsPrice':'2100', 'RowNo': '2', 'ColNo': '2'},  {'goodsId': '8806002007298','goodsName':'광동)비타500병180ml','goodsPrice':'1300', 'goodsCnt': '-1', 'RowNo': '3', 'ColNo': '3'}, {'goodsId': '8806002007298', 'goodsCnt': '1', 'goodsName':'광동)비타500병180ml','goodsPrice':'1300', 'RowNo': '4', 'ColNo': '4'}], "trDate": "20210101100000", "trNo": "1001" }

    response = {
    'resultCode': "000",
    'resultMsg': "SUCCESS",
    'data': data}
    return json.dumps(response, ensure_ascii=False)

@bp.route('/test_closed_2', methods=['POST'])
def test_closed_2():
    data = {"orderList":[{'goodsId': '8809350888168', 'goodsCnt': '-1', 'goodsName':'GET콜드브루아메리카노','goodsPrice':'2000', 'RowNo': '1', 'ColNo': '1'}, {'goodsId': '8809350888168', 'goodsCnt': '1', 'goodsName':'GET콜드브루아메리카노','goodsPrice':'2000', 'RowNo': '3', 'ColNo': '3'}], "trDate": "20210101100001", "trNo": "1004" }

    response = {
    'resultCode': "000",
    'resultMsg': "SUCCESS",
    'data': data}
    return json.dumps(response, ensure_ascii=False)

# http send (route_fridge, route_cigar)
def model_inference(dev_kind, companyId, storeId, deviceId, trade_info, work_user, barcode, needSalesInfo):
    try:
        if dev_kind == 'fridge':
            logger.info(f"model inf fridge")
            model_inference_url = f'http://{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_SERVER_HOST}:' + \
                                  f'{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_FRIDGE_SERVER_PORT}/tf_model'
        elif dev_kind == 'cigar':
            logger.info(f"model inf cigar")
            model_inference_url = f'http://{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_SERVER_HOST}:' + \
                                  f'{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_CIGAR_SERVER_PORT}/cigar_model'
        elif dev_kind == 'vaccine':
            logger.info(f"model inf vaccine")
            model_inference_url = f'http://{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_SERVER_HOST}:' + \
                                  f'{config_by_name[Config.BOILERPLATE_ENV].INFERENCE_VACCINE_SERVER_PORT}/vaccine_model'

        trNo, trade_date, trade_time, _ = trade_info
        trDate = datetime.combine(trade_date, trade_time).strftime('%Y-%m-%d_%H:%M:%S')
        trResponseDate = datetime.combine(trade_date, trade_time).strftime("%Y%m%d%H%M%S")
        headers = {'User-Agent':'Ubuntu 18.04.5 LTS'}
        res = requests.post(model_inference_url, headers=headers,
                            json={'companyId': companyId, 'storeId': storeId, 'deviceId': deviceId, 'trNo': trNo,
                                  'trDate': trDate, 'trResponseDate': trResponseDate, 'work_user': work_user, 'barcode':barcode, 'needSalesInfo':needSalesInfo},
                            verify=False, timeout=100)
        return res.json()
    except Exception as err:
        logger.error(f'[model_inference] requests error({str(err)})')
        logger.error(traceback.format_exc())
        abort(500, str(err))

@bp.route('/lets_infer', methods=['POST'])
def lets_infer(work_user = 'customer'):
    try:
        start = time.time()
        dao = DAO()
        log_dao = DAO(log=True)
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        try: needSalesInfo = bool(strtobool(request.json['needSalesInfo']))
        except KeyError: needSalesInfo = False
        try: companyId = request.json['companyId']
        except KeyError: companyId = dao.get_company_id(storeId, deviceId)

        try: barcode = request.json['barcode']
        except KeyError: barcode = 0
        # log 권한 요청 (냉장고, 담배, 주류, 백신)
        log, dev_kind = get_dev_stg_type(storeId, deviceId)

        trade_info = log_dao.insert_trade(companyId, storeId, deviceId)

        door_closed_result = model_inference(dev_kind, companyId, storeId, deviceId, trade_info, work_user, barcode, needSalesInfo)
        abort_ = door_closed_result['abort']

        # abort 처리.
        if abort_['code'] != 200:
            raise Exception(f"{abort_['msg']}")

        # 상품 정보 Return시 'abort' 내역 삭제
        del door_closed_result['abort']

        if not needSalesInfo:
            for order in door_closed_result['orderList']:
                order.pop('goodsName')
                order.pop('goodsPrice')

        dao.session.commit()
        return json.dumps(door_closed_result, ensure_ascii=False)
    except Exception as ex:
        dao.session.rollback()
        log.error(traceback.format_exc())
        log.exception(f'/lets_infer [{work_user}]')
        abort(int(abort_['code']), str(ex))
    finally:
        dao.session.close()
        log_dao.session.close()
        log.info(f'[lets_infer] total_time: {str(time.time() - start)[:4]}sec')


@bp.route('/release_src', methods=['POST'])
def release_src():
    response = {
        'resultCode': "000",
        'resultMsg': "SUCCESS",
        "data": {}
    }
    return json.dumps(response)

@bp.route('/admin_door_opened', methods=['POST'])
def admin_door_opened():
    return door_opened('admin')

@bp.route('/admin_door_closed', methods=['POST'])
def admin_door_closed():
    try:
        worker_type = request.json['worker']
    except Exception as err:
        logger.warning(f'request data안에 worker_type이 없습니다. worker => manager 로 변경합니다. ({str(err)})')
        worker_type = 'manager'

    return lets_infer(worker_type)

@bp.route('/door_alarm', methods=['POST'])
def door_alarm():
    try:
        dao = DAO()
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        requestTime = request.json['requestTime']
        try:
            companyId = request.json['companyId']
        except KeyError:
            companyId = dao.get_company_id(storeId, deviceId)

        if config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM == True:
            message = Alarm(companyId, storeId, deviceId)
            context = "- 디바이스 잠금장치(데드볼트) 오류입니다.\n"
            message.send_alimtalk("status_error", requestTime, context, "default")
        response = {
            'resultCode': "000",
            'resultMsg': "SUCCESS",
            "data": {}
        }
        
    except Exception as ex:
        dao.session.rollback()
        logger.error(traceback.format_exc())
        abort(500, str(ex))
        logger.exception('/door_alarm')
        response = {
            'resultCode': "001",
            'resultMsg': "FAIL, something wrong on device",
            "data": {}
        }
    finally:
        dao.session.close()
        return json.dumps(response)

@bp.route('/update_pog', methods=['POST'])
def update_pog():
    try:
        dao = DAO()
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        try:
            companyId = request.json['companyId']
        except KeyError:
            companyId = dao.get_company_id(storeId, deviceId)

        # log 권한 요청 (냉장고, 담배, 주류, 백신)
        log, dev_kind = get_dev_stg_type(storeId, deviceId)

        pogList = request.json['pogList']
        '''
        # 필드아이디 'pog' 구조:
        [
            {
                'goodsId': '8801234567890',
                'floor': 0,
                'column': 0,
            },
            {
                'design_infer_label': 'label_name',
                'floor': 0,
                'column': 1,
            },
            ...
        ]
        '''
        # '/update_pog'로 pog를 업데이트 하는 경우 goods_id와 design_pkey는 1:1 관계여야 함
        # goodsId에 해당하는 design_infer_label이 미리 정해져 있어야 함. 예를 들어 'sandwich1'
        # goods -> designs -> cells -> loadcells 테이블 순서로 update
        # dao = DAO()
        for pog in pogList:
            cell_pkey = dao.get_cell_pkey(companyId, storeId, deviceId, pog['floor'], pog['column'])
            design_pkey = dao.get_design_pkey(pog.get('goodsId', None), pog.get('design_infer_label', None))
            
            dao.delete_stocks(cell_pkey)
            dao.update_pog(cell_pkey, design_pkey)
            dao.insert_stocks(cell_pkey, design_pkey, 0)
        response = {
            'resultCode': "000",
            'resultMsg': "SUCCESS",
            "data": {}
        }
        return json.dumps(response)
    except Exception as ex:
        dao.session.rollback()
        log.error(traceback.format_exc())
        log.exception('/update_pog')
        abort(500, str(ex))
    finally:
        dao.session.close()

@bp.route('/regist_goods', methods=['POST'])
def regist_goods():
    try:
        dao = DAO()
        goodsId = request.json['goodsId']
        goodsName = request.json['goodsName']
        goodsMeanWeight = request.json['goodsMeanWeight']
        goodsStdWeight = request.json['goodsStdWeight']
        distLabel = request.json['distLabel']
        dao.insert_goods(goodsId, goodsName, distLabel, goodsMeanWeight, goodsStdWeight)
        response = {
            'resultCode': "000",
            'resultMsg': "SUCCESS",
            "data": {}
        }
        return json.dumps(response)
    except Exception as ex:
        dao.session.rollback()
        logger.error(traceback.format_exc())
        logger.exception('/regist_goods')
        abort(500, str(ex))
    finally:
        dao.session.close()


@bp.route('/status_info', methods=['POST'])
def status_info():
    try:
        dao = DAO()
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        try:
            companyId = request.json['companyId']
        except KeyError:
            companyId = dao.get_company_id(storeId, deviceId)


        status_info = get_status_info(companyId, storeId, deviceId)

        response = {
            "resultCode": "000",
            "resultMsg": "SUCCESS",
            "data": {"status_info" : status_info}
        }
        logger.info(status_info)

    except Exception as ex:
        # dao.session.rollback()
        logger.error(traceback.format_exc())
        response = {
                'resultCode': "001",
                'resultMsg': "FAIL, something wrong on device",
                "data": {}}
        
    finally:
        dao.session.close()
        return json.dumps(response, ensure_ascii=False)
        
# @bp.route('/just_test', methods=['POST'])
# def just_test():
#     try:
#         dao = DAO()
#         dao.insert_trade_log('0001', '00888', 's_00009', 0, 0, '8801037039993', '티오피)아메리카노200ml', 'top_americano_200_can',
#                              1, 216.2, 21.62, 271, 52, 0, 'customer', 'trade', '000')
#         response = {
#             'resultCode': "000",
#             'resultMsg': "SUCCESS",
#             "data": {}
#         }
#     except Exception as ex:
#         traceback.print_exc()
#         abort(500)
#         logger.exception('/just_test')

@bp.route('/qr_confirm', methods=['POST'])
def qr_confirm():
    response = {
        "resultCode" : "001",
        "resultMsg" : "FAIL",
        "userLevel" : None            
    }

    try:
        companyId = request.json['companyId']
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        qr_code = request.json['qr_code']
        message = Alarm(companyId, storeId, deviceId)
        customer_info_dict = re.get(f'{companyId}_{storeId}_customer_info')
        customer_info_dict = eval(customer_info_dict)
        if qr_code in list(customer_info_dict.keys()):
            response["userLevel"] = customer_info_dict[qr_code]["userLevel"]
            response["resultMsg"] = "SUCCESS"
            response["resultCode"] = "000"
            qr_name = customer_info_dict[qr_code]["name"]
            qr_userLevel = customer_info_dict[qr_code]["userLevel"]
            try:
                set_qr = {
                    "qr_code" : qr_code,
                    "name" : qr_name,
                    "userLevel" : qr_userLevel
                }
                re.set(f'{companyId}_{storeId}_{deviceId}_qr_confirm', str(set_qr))
            except Exception as err:
                logger.warning(traceback.format_exc())
                logger.warning(f'{companyId}_{storeId}_{deviceId}_qr_confirm set Error{err}')
        else:
            logger.warning(f"등록되지 않은 QR 코드입니다.")
            now = datetime.now()
            date_time = now.strftime('%Y-%m-%d_%H:%M:%S')
            message.send_slack(date_time, f"등록되지 않은 QR 코드입니다. {qr_code}",'vaccine')
            response["resultCode"] = "001"
    except requests.ConnectionError as e:
        logger.warning(f'[인터넷 연결 에러] {e}')
        response["resultCode"] = "002"
    except requests.RequestException as e:
        logger.warning(f'[요청 에러] {e}')
        response["resultCode"] = "003"
    except requests.Timeout as e:
        logger.warning(f'[타임아웃 에러(10초)] {e}')
        response["resultCode"] = "004"
        response["resultMsg"] = "FAIL"
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f'[Exception 에러] {e}')
    finally:
        return json.dumps(response)

@bp.route('/cigar_orderlist', methods=['POST'])
def cigar_orderlist():
    log, dev_kind = get_dev_stg_type(request.json['storeId'], request.json['deviceId'])
    dao = DAO()
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d_%H:%M:%S')
    response = {'result' : "SUCCESS"}
    try:
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        companyId = dao.get_company_id(storeId, deviceId)
        message = Alarm(companyId,storeId,deviceId)
        userId = request.json['userId']
        re.set(f'{companyId}_{storeId}_{deviceId}_ai_run',str(userId))
        tmpt = message.send_slack(date_time,f"[cigar_orderlist] 요청 성공 \n[파라미터]{json.loads(request.data)}",dev_kind)
    
    except Exception as err:
        message = Alarm(companyId,storeId,deviceId)
        tmpt = message.send_slack(date_time,f"[cigar_orderlist] 요청 실패 \n[파라미터]{json.loads(request.data)} \n ERROR{err}", dev_kind)
        log.error(traceback.format_exc())
        log.error(f'[cigar_orderlist] {err}')
        response['result'] = 'FAIL'
    return json.dumps(response, ensure_ascii=False)

@bp.route('/kakao_alarm', methods=['POST'])
def kakao_alarm():
    try:
        alarmHeader = request.json['alarmHeader']
        companyId = request.json['companyId']
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        alarmContext = request.json['alarmContext']
        try:
            floor = request.json['floor']
            floor_context = f'floor : {floor} 층 \n'
        except Exception as ex:
            logger.warning(f'floor is not defined {ex}')
            floor_context = ''
        finally:
            context = f'{floor_context}{alarmContext}'
        message = Alarm(companyId, storeId, deviceId)
        now = datetime.now()
        date_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        logger.info(context)
        message.send_alimtalk(alarmHeader, date_time, context, "interminds")
        response = {
            "resultCode": "000",
            "resultMsg": "SUCCESS"
        }
    except Exception as err:
        logger.error(traceback.format_exc())
        response = {
                'resultCode': "001",
                'resultMsg': "FAIL, something wrong on device"}  
    finally:
        return json.dumps(response)

@bp.route('/slack_alarm', methods=['POST'])
def slack_alarm():
    try:
        date_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        companyId = request.json['companyId']
        storeId = request.json['storeId']
        deviceId = request.json['deviceId']
        alarmContext = request.json['alarmContext']

        try:
            work_user = request.json['work_user']
        except KeyError:
            work_user = 'interminds'

        log, dev_kind = get_dev_stg_type(storeId, deviceId)

        try:
            message = Alarm(companyId, storeId, deviceId, work_user)
            message.send_slack(date_time, alarmContext, dev_kind)
        except Exception as err:
            log.error(f'Slack Message Send Error({str(err)})')
            response = {
                'resultCode': "002",
                'resultMsg': f"FAIL, {str(err)}"}
        else:
            response = {
            "resultCode": "000",
            "resultMsg": "SUCCESS"
        }
    except Exception as err:
        log.error(traceback.format_exc())
        response = {
                'resultCode': "001",
                'resultMsg': "FAIL, something wrong on device"}  
    finally:
        return json.dumps(response)

@bp.route('/')
@bp.route('/index')
def index():
    return '/' 

# @bp.before_request
# def limit_remote_addr():
#     if Config.BLACK_IP_BLOCKING == 'True':
#         if request.remote_addr in Config.BLACK_IP:
#             logger.error("[허용되지 않은 IP 차단] {}".format(request.remote_addr))
#             abort(403)

#     if Config.WHITE_IP_ALLOW == 'True':
#         if request.remote_addr not in Config.WHITE_IP:
#             logger.error("[허용되지 않은 IP 차단] {}".format(request.remote_addr))
#             abort(403)


def total_snapshot(dao, companyId, storeId, deviceId, work_user):
    # log 권한 요청 (냉장고, 담배, 주류, 백신)
    log, dev_kind = get_dev_stg_type(storeId, deviceId)

    floors = dao.get_shelf_floors(companyId, storeId, deviceId)
    for floor in floors:
        floor = floor[0]
        # log.info(f"{companyId}, {storeId}, {deviceId}, {floor}")
        log_str = snapshot(companyId, storeId, deviceId, floor)
        if log_str != "":
            log_msg = f'[total_snapshot] {companyId}_{storeId}_{deviceId}_f{floor}_loadcell : {log_str}'
            log.warning(log_msg)
            try:
                lc_alarm = Alarm(companyId, storeId, deviceId, work_user)
                now = datetime.now()
                date_time = now.strftime('%Y-%m-%d_%H:%M:%S')
                lc_alarm.send_slack(date_time, log_msg, dev_kind)
            except Exception as err:
                log.error(f'[total_snampshot] Slack Send Error({err})')


@bp.before_request
def before_request():
    root_logger.addHandler(logger_root.handlers[0])
    root_logger.addHandler(logger.handlers[0])
    if request.path == '/':
        root_logger.removeHandler(logger.handlers[0])
        logger_root.info(f'[요청 {request.path}] url: {request.url} | ip: {request.remote_addr} | {request.method}')
        try:
            logger_root.info(f'[요청 파라미터] {json.loads(request.data)}')
        except:
            pass
    else:
        root_logger.removeHandler(logger_root.handlers[0])
        if request.path == '/check_status' or request.path == '/release_src':
            dev_kind = 'check_status'
            log = LogDesignate(dev_kind)
        else:
            # log 권한 요청 (냉장고, 담배, 주류, 백신)
            log, dev_kind = get_dev_stg_type(request.json['storeId'], request.json['deviceId'])

        log.info(f'[{dev_kind}][요청 {request.path}] url: {request.url} | ip: {request.remote_addr} | {request.method}')
        try:
            log.info(f'[{dev_kind}][요청 파라미터] {json.loads(request.data)}')
        except:
            pass
# if __name__ == '__main__':
#     app.debug = True
#     # context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
#     # context.load_cert_chain("sr_cert.crt", "sr_key.key") 
#     app.run(host='0.0.0.0', port=5000)


# def align_center(log_str, max_len=100):
#     log_str = log_str.split('\n')
#     return '\n'.join([row.center(max_len, ' ') for row in log_str])

