import json
import requests
import logging
import logging.config
import ast
import time
import threading

from .data_access import DAO
from . import models
from .config import Config
from .config import config_by_name
import app.log_adapter
from .message.message import sendMany

logger = logging.getLogger('console_file')

# 알림 제목
subject_label = {
        "scope_error" : "오류 알림",
        "stock_alarm" : "재고 부족 알림",
        "status_error" : "하드웨어 오류 알림",
        "server_error" : "서버 오류 알림",
        "cpu_mem_warning" : "라즈베리 CPU사용량 경고 알림",
        "temper_warning" : "라즈베리 온도 경고 알림",
    }

class Alarm:
    def __init__(self, companyId, storeId, deviceId, work_user="admin"):
        self.dao = DAO()
        self.companyId = companyId
        self.storeId = storeId
        self.deviceId = deviceId
        self.company_name = self.dao.get_company_name(companyId)[0]
        self.store_name = self.dao.get_store_name(companyId, storeId)[0]
        self.device_pkey = self.dao.get_device_pkey(companyId, storeId, deviceId)
        self.device_alarm = self.dao.get_device_alarm(self.device_pkey)[0]
        self.work_user = work_user
        
    def __del__(self):
        pass
        #self.dao.session.close()
        
    def send_slack(self, date_time, message, dev_type='fridge'):
        if self.device_alarm == False:
            logger.info(f"[Slack] send, divice alarm : off")
            return
        url_list = Config.SLACK_URL[dev_type]
        date, time= date_time.split('_')
        final_msg = f"* 유저 모드 : {self.work_user}\n"+\
            f"* 날짜 : {date}\n"+\
            f"* 시간 : {time}\n"+\
            f"* 스토어 : {self.store_name}({self.storeId})\n"+\
            f"* 디바이스 : {self.deviceId}\n\n"+\
            f"* {message}"
        data = {'text': final_msg}
        for url in url_list:
            try:
                resp = requests.post(url=url, json=data, timeout=10)
            except Exception as e:
                logger.error(f'send url({url}) / msg({data})')
                logger.error(f'send_slack Error ({str(e)})')
        return resp
        
    def send_email(self, error_type, date_time, message, send_mode, header="", head_text=""):
        if self.device_alarm == False:
            logger.info(f"[Email] send {error_type}, divice alarm : off")
            return
        receiver = self._set_receiver("email", error_type, send_mode)
        if receiver == []:
            logger.info(f"[Email] send {error_type}, receiver : []")
            return
        subject = f"{self.store_name}점({self.storeId}) 냉장고 " + subject_label[error_type]
        context = self._set_email_template(error_type, subject, message, date_time, header, head_text)
        email_obj = {
            "toEmails": receiver,
            "subject": subject,
            "message": context
        }
        try:
            res = requests.post(url = Config.EMAIL_URL, data = json.dumps(email_obj))
            res.raise_for_status()
            logger.info(f"[Email] send {error_type} success, divice alarm : on")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"[Email] send {error_type} error", e.response.text)
            return False

    def _get_strlen(self, message, unit):
        '''
            한글 잘리지 않게 문자열 자를 위치 계산 
        '''
        str_len = 0
        if len(message.encode('utf-8')) > unit:
            split_idx = 0
            for char in message:
                str_len = str_len + len(char.encode('utf-8'))
                split_idx += 1
                if str_len > unit:
                    break
        else:
            return len(message)
        return split_idx

    def _split_message(self, message):
        '''
            1000바이트 이상이면 카톡 전송 되지 않음, 분할해서 전송
        '''
        unit = 800
        total_len = len(message)
        result = []
        while(1):
            if total_len < unit:
                result.append(message)
                break
            str_len = self._get_strlen(message, unit)
            result.append(message[:str_len])
            total_len -= str_len
            message = message[str_len:]
        return result

    def _send_msg(self, msg_list, error_type, receiver, date_time, header):
        for msg in msg_list:
            data = self._set_alimtalk_template(error_type, msg, receiver, date_time, header)
            res = sendMany(data)
            res.raise_for_status()
            time.sleep(2)

    def send_alimtalk(self, error_type, date_time, message, send_mode, header=""):
        '''
            error_type : scope_error, stock_alarm, status_error
            send_mode : interminds(인터마인즈 관리자에게만 전송), default(알림 켜져있는 모두에게 전송)
        '''
        if self.device_alarm == False:
            logger.info(f"[Alimtalk] send {error_type}, divice alarm : off")
            return
        receiver = self._set_receiver("alimtalk", error_type, send_mode)
        if receiver == []:
            logger.info(f"[Alimtalk] send {error_type}, receiver : []")
            return
        try:
            msg_list = self._split_message(message)
            send_th = threading.Thread(target=self._send_msg, args=(msg_list, error_type, receiver, date_time, header,))
            send_th.start()
            logger.info(f"[Alimtalk] send {error_type} success, divice alarm : on")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"[Alimtalk] send {error_type} error", e.response.text)
            return False
        except Exception as err:
            logger.error(f'[Alimtalk] send {error_type} error,{err}')
            return False

    def _set_receiver(self, alarm_type, error_type, send_mode):
        try:
            receiver_url = f"https://www.interminds-ssl.com:22445/get_receiver?alarmtype={alarm_type}"+\
                        f"&emailtype={error_type}&sendmode={send_mode}&companyid={self.companyId}&storeid={self.storeId}"
            res = requests.get(url=receiver_url)
            res.raise_for_status()
            return ast.literal_eval(res.text)
        except requests.exceptions.RequestException as e:
            logger.error("[Alarm] get_receiver error", e.response.text)
            return False
        except Exception as errr:
            logger.error("[Alarm] get_receiver error {errr}")
            return False

    def _set_email_template(self, error_type, subject, message, date_time, header="", head_text=""):
        message = message.replace('\n','<br>')
        context = f"""<html>
            <head>
                <h3>{header}</h3>
                <h3>[{date_time}] {head_text}</h3>
            </head>
            <body>
                <table style="fixed", border="1", cellpadding="10", cellspacing="1">
                        <tr style='line-height:200%'>
                            <h3>
                            &nbsp;&bull; companyId : {self.companyId} ({self.company_name}) <br>
                            &nbsp;&bull; storeId : {self.storeId} ({self.store_name}) <br>
                            &nbsp;&bull; deviceId : {self.deviceId}
                            </h3>
                        </tr>
                        <tr style='line-height:150%'>
                        {message}
                        </tr>
                </table>
                <br> <a href="https://interminds-gui.com">인터마인즈 관리자 페이지</a>
            </body>
        </html>"""
        return context

    def _set_alimtalk_template(self, error_type, message, receiver, date_time, header="", head_text=""):
        '''
            기존에 승인 받은 템플릿으로만 전송 가능, 템플릿 수정하고 싶으면 사전 승인 받아야함(1~3일 소요)
            알림톡은 하단 양식과 동일한 경우에만 전송되고, 동일하지 않은 경우 메시지로 대체 발송됨            

            [#{스토어 이름} #{오류코드} 알림]
            안녕하세요. 인터마인즈 입니다. 
            스마트 선반의 #{오류코드} 알림입니다.

            * 날짜 : #{날짜}
            * 시간 : #{시간}

            #{상품 위치} #{오류 명} 

            자세한 내용은 인터마인즈 관리자 페이지에서 확인 부탁드립니다.
        '''

        date, time = date_time.split('_')
        subject = f"[{self.store_name}{header} {subject_label[error_type]}]\n"
        context = "안녕하세요. 인터마인즈 입니다.\n"+\
                f"스마트 선반의 {subject_label[error_type]}입니다.\n\n"+\
                f"* 날짜 : {date}\n"+\
                f"* 시간 : {time}\n"+\
                f"* 스토어 : {self.store_name}({self.storeId})\n"+\
                f"* 디바이스 : {self.deviceId}\n\n"+\
                f"{message}\n"+\
                "자세한 내용은 인터마인즈 관리자 페이지에서 확인 부탁드립니다."
        data = {
            'messages': [
            {
                'to': receiver,
                'from': '03180397231',
                'text': subject+context,
                'kakaoOptions': {
                    'pfId': 'KA01PF210323063427423z6NhPzwHfOO',
                    'templateId': 'KA01TP210719070307052JVcYSkgheMz',
                    'buttons': [{
                        'buttonType': 'WL', # 웹링크
                        'buttonName': '관리자 페이지 이동',
                        'linkMo': 'https://interminds-gui.com',
                        'linkPc': 'https://interminds-gui.com'
                    }]
                }
            }]
        }
        return data
