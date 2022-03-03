import redis
import json
from keys import keys

from .config import Config, config_by_name

r = redis.Redis(host=config_by_name[Config.BOILERPLATE_ENV].REDIS_HOST, port=config_by_name[Config.BOILERPLATE_ENV].REDIS_PORT, \
    db=config_by_name[Config.BOILERPLATE_ENV].REDIS_DB, username=config_by_name[Config.BOILERPLATE_ENV].REDIS_USERNAME, \
    password=config_by_name[Config.BOILERPLATE_ENV].REDIS_PASSWORD,  charset=config_by_name[Config.BOILERPLATE_ENV].REDIS_CHARSET, \
    decode_responses=config_by_name[Config.BOILERPLATE_ENV].REDIS_DECODE_RESPONSES)


def snapshot(company_id, store_id, device_id, floor):
    try:
        from .data_access import DAO
        dao = DAO()
        log_str = ""
        lc_values = r.get('{}_{}_{}_f{}_loadcell'.format(company_id, store_id, device_id, floor))
        if lc_values is None:
            column_length = dao.get_cell_column_length(company_id, store_id, device_id, floor)
            r.set('{}_{}_{}_f{}_loadcell'.format(company_id, store_id, device_id, floor), json.dumps([0] * column_length))
            lc_values = r.get('{}_{}_{}_f{}_loadcell'.format(company_id, store_id, device_id, floor))
        # 로드셀 값 'str' -> '0' 치환
        #if 'NaN' in lc_values or 'Nan' in lc_values or '-Infinity' in lc_values:
            #log_str += lc_values
            #lc_values = lc_values.replace('NaN', '0').replace('Nan', '0').replace('-Infinity', '0')
            #log_str += " => " + str(lc_values)
        try:
            lc_values = eval(lc_values)
        except:
            log_str += lc_values
        r.set('{}_{}_{}_f{}_snapshot'.format(company_id, store_id, device_id, floor), json.dumps(lc_values))
    finally:
        # DB Session close
        dao.session.close()
        return log_str