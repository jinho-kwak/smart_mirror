"""
author: Jinho Ahn
e-mail: winarc24@interminds.ai
"""

import redis
import copy
import numpy as np
import pandas as pd
from .load_cell_logic_rel import Counter as relCounter
from .load_cell_logic_abs import Counter as absCounter
import json
import logging
import logging.config
# from .log_adapter import StyleAdapter
import app.log_adapter
from sqlalchemy.sql import text
from .data_access import DAO
import ast
from datetime import datetime as dt

from .config import Config, config_by_name

# from .log_getter import LogGetter
from .alarm import Alarm
from datetime import datetime
from .log_designate import LogDesignate
import app.util as util

r = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
    db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
    password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD, charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
    decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)


# Log 결정 Class
devkind = 'fridge'
log = LogDesignate(devkind)

class OrderList:
    def __init__(self, company_id, store_id, device_id, trade_no, trade_date, trade_time, infer_work_flag, work_user):
        self.company_id = company_id
        self.store_id = store_id
        self.device_id = device_id
        self.trade_no = trade_no
        self.trade_date = trade_date
        self.trade_time = trade_time
        self.dao = DAO()
        self.log_dao = DAO(log=True)
        self.set_config()
        self.message = Alarm(company_id, store_id, device_id, work_user)
        # 거래 (성공, 실패) 로그 저장.
        self.trade_log_dict = {}
        self.trade_log_dict['err_flag'] = False
        self.trade_log_dict['trade_log'] = []
        # 인퍼런스 실패 여부 판단 (True : 성공 / False : 실패)
        self.infer_work_flag = infer_work_flag

        self.date_str = trade_date + '_' + trade_time
        self.trDate = datetime.strptime(self.date_str, '%Y-%m-%d_%H:%M:%S')
        self.header = None

    def __del__(self):
        self.dao.session.close()
        self.log_dao.session.close()

    def set_config(self):
        self.device_pkey = self.dao.get_device_pkey(self.company_id, self.store_id, self.device_id)
        # set changes instance
        abs_permit_rate = eval(r.get(f'abs_permit_rate'))
        abs_permit_rate_else = float(r.get(f'abs_permit_rate_else'))
        rel_permit_rate = eval(r.get(f'rel_permit_rate'))
        rel_permit_rate_else = float(r.get(f'rel_permit_rate_else'))

        self.cells = self.dao.get_cells_by_device_pkey(self.device_pkey)
        self.Cs = [absCounter(c.design_pkey_master, self.dao, abs_permit_rate, abs_permit_rate_else) \
                    if c.load_cell_mode == 'abs' else relCounter(c.design_pkey_master, self.dao, rel_permit_rate, rel_permit_rate_else)\
                    for c in self.cells.itertuples()]

    def count(self, work_user):
        # set master dataframe
        master = self.dao.get_loadcells_with_floor_by_device_pkey(self.device_pkey)
        floors = master['shelf_floor'].unique()
        lc_cali_list = list(master['loadcell_column'])

        # lc_cali 가 없으면 만들어낸다.
        make_lc_cali = {}
        W_lc_cali = r.get(f'{self.company_id}_{self.store_id}_{self.device_id}_lc_cali')
        if W_lc_cali:
            W_lc_cali = ast.literal_eval(W_lc_cali)
        else:
            log.info(f'{self.company_id}_{self.store_id}_{self.device_id}_lc_cali 를 생성합니다.')
            for f in floors:  
                ints_to_string = [str(int) for int in lc_cali_list[:list(master['shelf_floor']).count(f)]]
                dictionary = {string:1 for string in ints_to_string}
                make_lc_cali[str(f)] = dictionary
                del lc_cali_list[:list(master['shelf_floor']).count(f)]

            r.set(f'{self.company_id}_{self.store_id}_{self.device_id}_lc_cali', str(make_lc_cali))
            W_lc_cali = r.get(f'{self.company_id}_{self.store_id}_{self.device_id}_lc_cali')
            W_lc_cali = ast.literal_eval(W_lc_cali)

        
        # add columns about weights
        w_o = []  # 문이 닫힐 때 로드 셀 무게
        w_c = []  # 문이 열릴 때 로드 셀 무게
        w_cali = []
        
        for f in floors:
            log_msg = f"[lc_inf] {self.company_id}_{self.store_id}_{self.device_id}_f{f}_"
            w_cali += list(W_lc_cali[str(f)].values())
            W_o = r.get(f'{self.company_id}_{self.store_id}_{self.device_id}_f{f}_snapshot')
            if W_o:
                try:
                    w_o += ast.literal_eval(W_o)
                except:
                    try:
                        log_msg += f"snapshot : {W_o}"
                        log.warning(log_msg)
                        self.message.send_slack(self.date_str, log_msg)
                    except Exception as err:
                        log.error(f'[lc_inf] Slack Send Error({err})')
            else:
                r.set(f'{self.company_id}_{self.store_id}_{self.device_id}_f{f}_snapshot', str([0] * (master['loadcell_column'].max() + 1)))
                w_o += [0] * (master['loadcell_column'].max() + 1)
            W_c = r.get(f'{self.company_id}_{self.store_id}_{self.device_id}_f{f}_loadcell')
            if W_c:
                try:
                    w_c += ast.literal_eval(W_c)
                except:
                    try:
                        log_msg += f"loadcell : {W_c}"
                        log.warning(log_msg)
                        self.message.send_slack(self.date_str, log_msg)
                    except Exception as err:
                        log.error(f'[lc_inf] Slack Send Error({err})')
            else:
                r.set(f'{self.company_id}_{self.store_id}_{self.device_id}_f{f}_loadcell', str([0] * (master['loadcell_column'].max() + 1)))
                w_c += [0] * (master['loadcell_column'].max() + 1)

        try:
            assert(len(w_c) == len(w_cali)) ## columns 수가 다르면 error
            for i in range(len(w_cali)):
                if 0 < w_cali[i] < 2:
                    w_o[i] = round(w_o[i] * w_cali[i], 1)
                    w_c[i] = round(w_c[i] * w_cali[i], 1)     
                else:
                    log.error(f'*****lc_cali : {w_cali[i]} value error*****')
                    raise(ValueError)
        except AssertionError:
            raise Exception('*****lc_cali의 갯수와 columns의 갯수가 맞지 않습니다*****')
        
        master['w_o'] = w_o
        master['w_c'] = w_c
        # sum weights by each cell
        master = master.groupby(['cell_pkey', 'shelf_pkey', 'shelf_floor'], as_index=False).sum()

        # merge master and cells table
        master = pd.merge(self.cells, master, on=['cell_pkey', 'shelf_pkey'])

        # get model_name
        model_name_list = [self.dao.get_model_name_from_shelf(self.device_pkey, m.shelf_floor) for m in master.itertuples()]

        # inference mode 설정된 pog 강제 'lc' 전환 / 설정된 pog가 아닐 경우 설정된 inference_mode 반영
        lc_mode_list = ['lc' if self.dao.get_lc_mode_goods_list(cell[0].design_pkey_master, cell[1]) else cell[0].inference_mode for cell in zip(self.cells.itertuples(), model_name_list)]
        master['change_infer_mode'] = lc_mode_list

        # infer 결과 실패할 경우 all 'lc' mode 강제 전환
        if not self.infer_work_flag:
            log.warning("[lc_inf] inference logic Fail > (ALL) 'lc' mode convert!!\n")
            must_lc_mode_list = ['lc' for i in range(len(master))]
            master['change_infer_mode'] = must_lc_mode_list

        # add 'stock' column
        stock_list = [self.dao.get_stocks(m.cell_pkey) for m in master.itertuples()]
        master['stock'] = stock_list
        # master['cell_pkey', 'shelf_pkey', 'design_pkey_master', 'design_pkey_front', 'cell_column', 
        #        'stock_count_max', 'inference_mode', 'load_cell_mode', 'shelf_floor', 'w_o', 'w_c', 'stock']

        # get vision inference results
        design_pkey_inf_main = []
        design_pkey_inf_empty = []
        for f in floors:
            design_pkey_inf_main += ast.literal_eval(r.get('{}_{}_{}_f{}_inf_main'.format(self.company_id, self.store_id, self.device_id, f)))
            design_pkey_inf_empty += ast.literal_eval(r.get('{}_{}_{}_f{}_inf_empty'.format(self.company_id, self.store_id, self.device_id, f)))
            # design_pkeys = ast.literal_eval(r.get('{}_{}_{}_f{}_inf_result'.format(self.company_id, self.store_id, self.device_id, f)))
            # design_pkey_front_c += list(map(lambda x: None if x == -1 else int(x), design_pkeys))
        try:
            master['design_pkey_inf_main'] = design_pkey_inf_main
            master['design_pkey_inf_empty'] = design_pkey_inf_empty
        except ValueError as ve:
            log.error(f'[LC]c:{self.company_id}/s:{self.store_id}/d:{self.device_id}: [에러] Vision inference 결과의 수와 Device 내의 전체 컬럼 수가 같지 않습니다.')
            raise ve
            
        # get cell_alert_dict
        cell_alert_dict = r.get('{}_{}_{}_cell_alert'.format(self.company_id, self.store_id, self.device_id))
        if cell_alert_dict:
            cell_alert_dict = ast.literal_eval(cell_alert_dict)
        else:
            cell_alert_dict = {}
            for floor in floors:
                cell_alert_dict[str(floor)] = {}
                
        order_list = []
        out_order_list = []    # 출고 리스트
        in_order_list = []     # 입고 리스트
        goods_log = [[] for f in floors]
        weight_log = [[] for f in floors]
        logic_type_log = [[] for f in floors]
        no_change = [True]    
        error = [False]
        
        try:
            master['total_floor'] = [len(floors) for i in range(len(master['cell_pkey']))]
            temp_goods_info = list()
            for dpkey in master['design_pkey_master']:
                temp_goods_info.append(self.dao.get_goods_by_design_pkey(dpkey))
            master['goods_info'] = temp_goods_info
        except Exception as e:
            log.error(f"Get total floor from DB Error [{e}]")

        def compute(m, c):
            
            msg = str() # 강제 lc 변환 slack log 저장
            # mode별 분기
            kwargs = {
                'weight_at_open': m.w_o,
                'weight_at_close': m.w_c,
                'stock_at_open': m.stock,
                'design_pkey_at_open': m.design_pkey_front,
                'design_pkey_inf_main': m.design_pkey_inf_main,
                'design_pkey_inf_empty': m.design_pkey_inf_empty,
                'inference_mode': m.change_infer_mode,
                'empty_mode': m.empty_mode if m.empty_mode else 'cv',  
            }
            # 강제 inference_mode 변환 로그. (mix -> lc) / 인퍼런스 서버 다운으로 인한 강제 'lc' 전환은 출력 X
            if (m.change_infer_mode != m.inference_mode) and (self.infer_work_flag == True):
                log.warning(f"[inference_mode] cell_pkey({m.cell_pkey}) / " + \
                            f"shelf_floor({m.shelf_floor})|cell_column({m.cell_column})|design_pkey({m.design_pkey_inf_main}) " + \
                            f"강제 '{m.inference_mode}' -> '{m.change_infer_mode}' 전환")
                
                try:
                    # m.goods_info
                    # [0] -> goods_id / [1] -> goods_name / [2] -> design_infer_label  
                    msg += f"{m.total_floor - m.shelf_floor}층 {m.cell_column + 1}번째 칸 인퍼런스 모드 확인 요망\n" \
                            f"(해당 상품은 비젼인식이 불가하여, 인퍼런스 모드 'mix' -> 'lc'로 강제 전환하여 동작하였습니다.)\n"\
                            f"POG : {m.goods_info[1]}({m.goods_info[0]})\n[관리자 페이지]인퍼런스 모드 'mix' -> 'lc' 전환이 필요합니다.\n\n"             
                except Exception as e:
                    msg = ""
                    log.error(f"강제 인퍼런스 'lc' 모드 전환 상품 리스트 슬랙 알림 메세지 생성 에러 [{e}]")

            status, admin_log_info = c.count(**kwargs)
            if not status:
                self.trade_log_dict['err_flag'] = True

            logic_type_log[int(m.shelf_floor)].append(f'{m.load_cell_mode}/{m.change_infer_mode}/{m.empty_mode}')
            if m.load_cell_mode == 'abs':
                w_o = None
                weight_log[int(m.shelf_floor)].append(f'{int(m.w_c)}')
            elif m.load_cell_mode == 'rel':
                w_o = m.w_o
                weight_log[int(m.shelf_floor)].append(f'{int(m.w_c)}({int(w_o - m.w_c)})')
            if m.change_infer_mode == 'mix':
                # update design_pkey_front
                design_pkey_front_c = None if c.empty else m.design_pkey_inf_main
                self.dao.update_design_pkey_front(m.cell_pkey, design_pkey_front_c)

            # 남은 재고 관련
            stock_goods_id_value = {}
            stock_goods_name_value = {}
            if self.dao.check_stocks(m.cell_pkey):
                self.dao.delete_stocks(m.cell_pkey)
            for key, value in c.stock.items():
                goods_rst = self.dao.get_goods_by_design_pkey(key)
                goods_id, goods_name = goods_rst.goods_id, goods_rst.goods_name
                stock_goods_id_value[goods_id] = stock_goods_id_value.get(goods_id, 0) + value
                stock_goods_name_value[goods_name] = stock_goods_name_value.get(goods_name, 0) + value
                self.dao.insert_stocks(m.cell_pkey, key, value)

            # 거래 상품 관련
            # trade_goods_id_value = {}  # for orderlist
            trade_goods_name_value = {}  # for log
            for key, value in c.change.items():
                no_change[0] = False
                goods_rst = self.dao.get_goods_by_design_pkey(key)
                goods_id = goods_rst.goods_id
                goods_name = goods_rst.goods_name
                design_rst = self.dao.get_designs_by_design_pkey(key)
                goods_label = design_rst.design_infer_label
                goods_mean_weight = design_rst.design_mean_weight
                goods_std_weight = design_rst.design_std_weight
                goods_price = self.dao.get_sale_price(self.store_id, key)
            
                # trade_goods_id_value[goods_id] = trade_goods_id_value.get(goods_id, 0) + value
                trade_goods_name_value[goods_name] = trade_goods_name_value.get(goods_name, 0) + value
                order_dict = {
                    'goodsId': goods_id,
                    'goodsName': goods_name,
                    'goodsCnt': str(value),
                    'goodsPrice': None if goods_price is None else str(goods_price),
                    'RowNo': str(m.shelf_floor),
                    'ColNo': str(m.cell_column)
                }

                log_order_dict = copy.deepcopy(order_dict)

                order_list.append(order_dict)

                log_order_dict['goods_mean_weight'] = None if goods_mean_weight is None else str(goods_mean_weight)
                log_order_dict['Cell_Pkey'] = None if m.cell_pkey is None else str(m.cell_pkey)

                if value > 0:   # (출고)
                    out_order_list.append(log_order_dict)
                elif value < 0:   # (입고)
                    in_order_list.append(log_order_dict)
                
                # 거래 로그 저장 (강제 status = '000' 정상화)
                temp_trade_log = {}
                temp_trade_log['trade_log_no'] = self.trade_no
                temp_trade_log['trade_log_date'] = self.trade_date
                temp_trade_log['trade_log_time'] = self.trade_time
                temp_trade_log['company_id'] = self.company_id
                temp_trade_log['store_id'] = self.store_id
                temp_trade_log['device_id'] = self.device_id
                temp_trade_log['shelf_floor'] = m.shelf_floor
                temp_trade_log['cell_column'] = m.cell_column
                temp_trade_log['goods_id'] = goods_id
                temp_trade_log['goods_name'] = goods_name
                temp_trade_log['goods_label'] = goods_label
                temp_trade_log['goods_count'] = value
                temp_trade_log['stock_left'] = str(stock_goods_id_value)
                temp_trade_log['goods_mean_weight'] = goods_mean_weight
                temp_trade_log['goods_std_weight'] = goods_std_weight
                temp_trade_log['open_weight'] = w_o
                temp_trade_log['close_weight'] = m.w_c
                temp_trade_log['duration'] = None
                temp_trade_log['work_user'] = work_user
                temp_trade_log['work_type'] = 'trade'
                temp_trade_log['sale_price'] = goods_price
                temp_trade_log['status_code'] = '000'

                self.trade_log_dict['trade_log'].append(temp_trade_log)
                        
            goods_log[int(m.shelf_floor)].append(trade_goods_name_value)

            # error 여부
            if not status:
                error[0] = True
                cell_alert_dict[str(m.shelf_floor)][str(m.cell_column)] = 1
                log.error(f'[LC]c:{self.company_id}/s:{self.store_id}/d:{self.device_id}/f:{m.shelf_floor}/c:{m.cell_column}/inference_mode:{m.change_infer_mode}/load_cell_mode:{m.load_cell_mode}/open_weight:{w_o}/close_weight:{m.w_c}/change:{trade_goods_name_value}/stock:{stock_goods_name_value}')
                goods_log[int(m.shelf_floor)][-1] = 'error'
            else:
                cell_alert_dict[str(m.shelf_floor)][str(m.cell_column)] = 0
                
            return msg

        def alarm(m,c):
            result = ""
            if m.change_infer_mode == 'mix' and m.empty_mode == 'cv':
                design_pkey_front_c = None if c.empty else m.design_pkey_inf_main
                
                # need reverse floor
                max_floor = self.dao.get_max_floor(self.device_pkey)
                floors = max_floor-m.shelf_floor+1
                columns = m.cell_column+1

                # 엠티 아니고, pog랑 비전 결과가 다르고, 이전 비전 현재 비전 다른 경우
                # if m.design_pkey_inf_empty and design_pkey_front_c != m.design_pkey_master and design_pkey_front_c != m.design_pkey_front:
                #     m_goods = self.dao.get_goods_by_design_pkey(m.design_pkey_master)
                #     now_goods = self.dao.get_goods_by_design_pkey(design_pkey_front_c)
                #     pre_goods = self.dao.get_goods_by_design_pkey(m.design_pkey_front)
                #     pre_goods_detail = f"{pre_goods.goods_name}({pre_goods.goods_id})" if pre_goods else "empty"
                #     result = f"- {floors}층 {columns}칸 비전 확인 요망\n"+\
                #                 f"POG : {m_goods.goods_name}({m_goods.goods_id})\n"+\
                #                 f"비전 결과 : {now_goods.goods_name}({now_goods.goods_id})\n"+\
                #                 f"이전 거래 비전 결과 : {pre_goods_detail}\n\n"

                # 로드셀이 20g 이하인데 비전 결과가 엠티가 아닌 경우
                if m.design_pkey_inf_empty and m.w_c <= 20 and m.load_cell_mode == 'abs':
                    now_goods = self.dao.get_goods_by_design_pkey(design_pkey_front_c)
                    m_goods = self.dao.get_goods_by_design_pkey(m.design_pkey_master)
                    result = f"- {floors}층 {columns}칸 비전 확인 요망\n"+\
                                "(로드셀 20g 이하, 비전 결과 엠티 아님)\n"+\
                                f"POG : {m_goods.goods_name}({m_goods.goods_id})\n"+\
                                f"비전 결과 : {now_goods.goods_name}({now_goods.goods_id})\n"+\
                                f"로드셀 무게 : {m.w_c}g\n\n"

                # 로드셀이 20g 초과인데 비전 결과가 엠티인 경우             
                elif not m.design_pkey_inf_empty and m.w_c > 20 and m.load_cell_mode == 'abs':
                    m_goods = self.dao.get_goods_by_design_pkey(m.design_pkey_master)
                    result = f"- {floors}층 {columns}칸 비전 확인 요망\n"+\
                                "(로드셀 20g 이상, 비전 결과 엠티)\n"+\
                                f"POG : {m_goods.goods_name}({m_goods.goods_id})\n"+\
                                f"비전 결과 : empty\n"+\
                                f"로드셀 무게 : {m.w_c}g\n\n"
            return result
        
        try:
            alarm_msg = ""
            # 로그 내용 저장 (에러 여부 판단 초기화)
            self.trade_log_dict['trade_log'] = []
            self.trade_log_dict['err_flag'] = False
            
            mix_to_lc_msg = ""
            for m, c in zip(master.itertuples(), self.Cs):
                mix_to_lc_msg += compute(m, c)
                alarm_msg += alarm(m,c)

            # DB 거래 로그 저장.
            for trade in self.trade_log_dict['trade_log']:
                # 한번이라도 거래 에러시 모든 거래 001(에러)처리
                if self.trade_log_dict['err_flag']:
                    trade['status_code'] = '001'
                # db 로그테이블 입력
                self.log_dao.insert_trade_log(
                    trade_log_no=trade['trade_log_no'],
                    trade_log_date=trade['trade_log_date'],
                    trade_log_time=trade['trade_log_time'],
                    company_id=trade['company_id'],
                    store_id=trade['store_id'],
                    device_id=trade['device_id'],
                    shelf_floor=trade['shelf_floor'],
                    cell_column=trade['cell_column'],
                    goods_id=trade['goods_id'],
                    goods_name=trade['goods_name'],
                    goods_label=trade['goods_label'],
                    goods_count=trade['goods_count'],
                    stock_left=trade['stock_left'],
                    goods_mean_weight=trade['goods_mean_weight'],
                    goods_std_weight=trade['goods_std_weight'],
                    open_weight=trade['open_weight'],
                    close_weight=trade['close_weight'],
                    duration=trade['duration'],
                    work_user=trade['work_user'],
                    work_type=trade['work_type'],
                    sale_price=trade['sale_price'],
                    status_code=trade['status_code'])

            # Header Define
            if work_user == 'manager':
                self.header = "(관리자 모드)"
            elif work_user == 'interminds':
                self.header = "(테스트 모드)"
            else:
                self.header = "(사용자 모드)"

            # Alarm msg send
            if config_by_name[Config.BOILERPLATE_ENV].EMAIL_ALARM == True and alarm_msg != "":
                try:
                    #self.message.send_alimtalk("scope_error", self.date_str, alarm_msg, "interminds", header)
                    resp = self.message.send_slack(self.date_str, alarm_msg)
                    log.info(f'[Alarm Msg Slack] Send Success.')
                except Exception as err:
                    log.error(f'[Alarm Msg Slack] send Error({err}) : {alarm_msg}')
            
            if no_change[0]:
                self.log_dao.insert_trade_log(
                    trade_log_no=self.trade_no,
                    trade_log_date=self.trade_date,
                    trade_log_time=self.trade_time,
                    company_id=self.company_id,
                    store_id=self.store_id,
                    device_id=self.device_id,
                    shelf_floor=None,
                    cell_column=None,
                    goods_id=None,
                    goods_name=None,
                    goods_label=None,
                    goods_count=None,
                    stock_left=None,
                    goods_mean_weight=None,
                    goods_std_weight=None,
                    open_weight=None,
                    close_weight=None,
                    duration=None,
                    work_user=work_user,
                    work_type='no_work',
                    sale_price=None,
                    status_code='001' if error[0] else '000')

            self.log_dao.session.commit()

        except Exception as e:
            self.log_dao.session.rollback()
            raise e

        finally:
            self.log_dao.session.close()
        
        r.set('{}_{}_{}_cell_alert'.format(self.company_id, self.store_id, self.device_id), str(cell_alert_dict))
        
        # logging
        total_log = {}
        err_catch = False
        
        idx = 0
        for row in goods_log:
            for col in row:
                # error process
                if col == 'error':
                    err_catch = True
                else:
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
                
        #if err_catch:
        #    raise Exception('lc_inf error occured')
        
        r.set('{}_{}_orderlist'.format(self.store_id, self.device_id), json.dumps({'orderList': order_list}))
        
        # 층 설정
        floor_list = [str(f'{(len(logic_type_log) - floor)}') for floor in range(len(logic_type_log))]

        # 상품 판매 정보 오류 (상품이 복수 개 표현 될 때, 카톡 알람)
        try:
            # 인퍼런스모드 강제 mix -> lc 알림 메세지
            try:
                if len(mix_to_lc_msg) > 0:
                    self.message.send_slack(self.date_str, mix_to_lc_msg)
            except Exception as e:
                log.error(f"인퍼런스 모드 강제 mix -> lc 슬랙 알림 에러 {e}")
                
            result_msg = ""
            # floor_goods_info[0] : 판매 상품 정보 / floor_goods_info[1] : 각 층
            for floor_goods_info in zip(goods_log, floor_list):
                column = 1
                floor = floor_goods_info[1]
                # 각 층 판매 상품 정보 확인
                for goods in floor_goods_info[0]:
                    if len(goods) > 1:
                        log.warning(f'floor : {floor}층 / column : {column}칸 / 상품 판매 정보 확인 필요 : {goods}')
                        result = f"- {floor}층 {column}칸 상품 판매 로그 확인 요망\n"+\
                                f"판매 결과 : {goods}\n\n"

                        result_msg += result
                    column += 1
            
            if result_msg != "":
                # Kakao / Slack Msg Send
                try:                    
                    #self.message.send_alimtalk("scope_error", self.date_str, result_msg, "interminds", self.header)
                    resp = self.message.send_slack(self.date_str, result_msg)
                    log.info(f'[Alarm Msg] (상품 판매 정보 오류) Send Success.')
                except Exception as exx:  # 카톡 알림 발송 오류
                    log.error(f'[kakaotalk] error {exx}')

        except Exception as e:
            log.error(f'[Sale Goods Info Alarm Talk Error] {str(e)}')
        
        # log
        log_str = """
\n- Loadcell goods in & out -

------------ logic type ------------
(loadcell mode/inference mode/empty mode)

""" + "\n".join([f'[{log_[1]}층]{log_[0]}' for log_ in zip(list(map(lambda row: " | ".join([f"#{i} "+f"{cell}".center(8, " ") for (i, cell) in enumerate(row)]), logic_type_log)), floor_list)]) + """

------------ weight ------------
""" + "\n".join([f'[{log_[1]}층]{log_[0]}' for log_ in zip(list(map(lambda row: " | ".join([f"#{i} "+f"{cell}".center(8, " ") for (i, cell) in enumerate(row)]), weight_log)), floor_list)]) + """

------------ goods -------------
""" + "\n".join([f'[{log_[1]}층]{log_[0]}' for log_ in zip(list(map(lambda row: " | ".join([f"#{i} "+f"{cell if cell else ''}".center(8, " ") for (i, cell) in enumerate(row)]), goods_log)), floor_list)]) + f"""

------------ total -------------
{total_log_d if total_log_d else 'nothing in & out'}

---------- order list ----------
"""
        if order_list:
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
            
        util.LogGetter.log += util.align_center(log_str)


        """
        log_str = '[LC]c:'+str(self.company_id)+'/s:'+str(self.store_id)+'/d:'+str(self.device_id)+'/goods in & out'
        log_str += '\n/trade_no:'+str(self.trade_no)+'/trade_date:'+str(self.trade_date)+'/trade_time:'+str(self.trade_time)
        log_str += '\n'+'- weight -'.center(100, ' ')+'\n' 
        log_str += '\n'.join(list(map(lambda row: ' | '.join(row), weight_log)))
        log_str += '\n'+'- goods -'.center(100, ' ')+'\n'
        log_str += '\n'.join(list(map(str, goods_log)))
        log_str += '\n'+'- total -'.center(100, ' ')
        log_str += '\n' + str(total_log_d)
        log_str += '\n'+'- order_list -'.center(100, ' ')
        log_str += '\n' + str(order_list)
        """

        # add pandas column 'ic_inf_log' for admin_error_log
        lc_inf_weight = []
        lc_inf_count = []
        lc_inf_permit_rate = []
        lc_inf_permit_else_rate = []
        for c in self.Cs:
            lc_inf_weight.append(c.ADMIN_LOG_WEIGHT)
            lc_inf_count.append(c.ADMIN_LOG_COUNT)
            lc_inf_permit_rate.append(c.PERMIT_RATE)
            lc_inf_permit_else_rate.append(c.PERMIT_ELSE_RATE)
        master['lc_inf_weight'] = lc_inf_weight
        master['lc_inf_count'] = lc_inf_count
        master['lc_inf_permit_rate'] = lc_inf_permit_rate
        master['lc_inf_permit_else_rate'] = lc_inf_permit_else_rate

        return (cell_alert_dict, {'orderList': order_list}, log_str, master)

# save error img
def save_error_img(self):
    pass

# def align_center(log_str, max_len=100):
#     log_str = log_str.split('\n')
#     return '\n'.join([row.center(max_len, ' ') for row in log_str])

