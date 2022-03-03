import redis
import logging.config
import logging
import ast
import traceback

from .data_access import DAO
from .config import Config
from .config import config_by_name


logger = logging.getLogger('basic_log')

re = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
    db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
    password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD, charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
    decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)

def get_status_info(companyId, storeId, deviceId):
    dao = DAO()
    try:
        # get status_check_dict
        re_status_check = re.get(f'{companyId}_{storeId}_{deviceId}_status_check')
        if re_status_check:
            re_status_check = ast.literal_eval(re_status_check)
        else:
            re_status_check = {}

        # get cell_alert_dict
        re_cell_alert = re.get(f'{companyId}_{storeId}_{deviceId}_cell_alert')
        if re_cell_alert:
            re_cell_alert = ast.literal_eval(re_cell_alert)
        else:
            raise Exception(f'Key of {companyId}_{storeId}_{deviceId}_cell_alert does not exist in Reddies.')
            

        status_info = []

        d_storage_type = dao.get_device_type(companyId, storeId, deviceId)
        if d_storage_type == "CG":
            df_pog_goods = dao.get_ciga_pog_goods(companyId, storeId, deviceId)
        else:
            df_pog_goods = dao.get_pog_goods(companyId, storeId, deviceId)

        for c_floor in re_cell_alert:
            c_status_info = {}
            c_status_info['floor'] = c_floor
            cells_info = re_cell_alert[c_floor]


            cells_list = []
            for cell_no, cell_status in cells_info.items():
                cell_dict = {}
                cell_dict['cellNo'] = cell_no
                cell_dict['status'] = cell_status
                
                df_cell_filter = df_pog_goods[(df_pog_goods['shelf_floor'] == int(c_floor)) \
                                        & (df_pog_goods['cell_column'] == int(cell_no))]
                goodsNameKr = df_cell_filter['goods_name'].tolist()[0]
                goodsId = df_cell_filter['goods_id'].tolist()[0]
                cell_dict['goodsNameKr'] = goodsNameKr
                cell_dict['goodsId'] = goodsId
                cells_list.append(cell_dict)
            c_status_info['cells'] = cells_list

            c_status_info['loadcell'] = re_status_check[c_floor]['lc']
            c_status_info['network'] = re_status_check[c_floor]['network']
            c_cams = re_status_check[c_floor]['cam']

            cams_list = []
            for cam_name, cam_status in c_cams.items():
                cam_dict = {}
                cam_dict['name'] = cam_name
                cam_dict['status'] = cam_status
                cams_list.append(cam_dict)

            c_status_info['cam'] = cams_list
            status_info.append(c_status_info)
    except Exception as ex:
        # dao.session.rollback()
        traceback.print_exc()
        
    finally:
        dao.session.close()
        return status_info
