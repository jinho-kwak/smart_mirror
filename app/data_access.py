#-*-coding utf-8-*-
"""
author: Jinho Ahn
e-mail: winarc24@interminds.ai
"""

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.sql import text, case
import json
# import models
from . import models
from keys import keys
import math
from .config import Config
from .config import config_by_name

engine = create_engine(config_by_name[Config.BOILERPLATE_ENV].SQLALCHEMY_DATABASE_URI)
# engine = create_engine(f'postgresql://postgres:{keys.get("postgres", "./keys")}@sr-test-1.ctnphj2dxhnf.ap-northeast-2.rds.amazonaws.com:5432/emart24')
Session = scoped_session(sessionmaker(bind=engine))
LogSession = scoped_session(sessionmaker(bind=engine))
metadata = MetaData(bind=engine)

class DAO:
    def __init__(self, log=False):
        if log:
            self.session = LogSession()
        else:
            self.session = Session()
        self.trade_log = Table('trade_log', metadata, autoload=True)
        self.cigar_trade_log = Table('cigar_trade_log', metadata, autoload=True)
        self.vaccine_trade_log = Table('vaccine_trade_log', metadata, autoload=True)
        self.vaccine_trade_check = Table('vaccine_trade_check', metadata, autoload=True)
        self.vaccine_trade_pog = Table('vaccine_trade_pog', metadata, autoload=True)

    # route_main.py - 신세계측 API 요청 시 companyId가 포함되지 않은 경우 예외처리
    def get_company_id(self, store_id, device_id=None):
        company_id = self.session.query(models.Company.company_id)\
            .select_from(models.Device)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .join(models.Partner, models.Partner.partner_pkey==models.Company.partner_pkey)\
            .filter(models.Partner.partner_id == '001',
                    models.Store.store_id == store_id)
        if device_id is not None:
            company_id = company_id.filter(models.Device.device_id == device_id)
        company_id = company_id.first().company_id
        return company_id


    def get_device_shelf(self, company_id, store_id, device_id):
        """
        SELECT 
        DEV.DEVICE_PKEY, DEV.DEVICE_ID, DEV.DEVICE_INSTALL_TYPE, DEV.DEVICE_STORAGE_TYPE, DEV.OPERATION, DEV.STORE_PKEY, DEV.ALARM,
        SH.SHELF_PKEY, SH.DEVICE_PKEY, SH.SHELF_FLOOR, SH.SHELF_STORAGE_TYPE, SH.MODEL_PKEY,
        MO.MODEL_NAME AS SHELF_MODEL
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, MODELS MO
        WHERE COM.COMPANY_ID = {company_id}
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = {store_id}
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = {device_id}
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.MODEL_PKEY = MO.MODEL_PKEY
        ORDER BY SH.SHELF_FLOOR
        """
        query = self.session.query(models.Device, models.Shelf, models.Model.model_name.label('shelf_model'))\
            .select_from(models.Shelf)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .join(models.Model, models.Model.model_pkey==models.Shelf.model_pkey)\
            .filter(
                models.Company.company_id==company_id,
                models.Store.store_id==store_id,
                models.Device.device_id==device_id
            ).order_by(models.Shelf.shelf_floor)
        df = pd.read_sql(query.statement, self.session.bind)
        return df

    # route_main.py에서 device의 floor 개수를 가져오는 용도
    def get_shelf_floors(self, company_id, store_id, device_id):
        floors = self.session.query(models.Shelf.shelf_floor)\
            .select_from(models.Shelf)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(
                models.Company.company_id==company_id,
                models.Store.store_id==store_id,
                models.Device.device_id==device_id
            ).order_by(models.Shelf.shelf_floor).all()
        return floors

    # route_inference.py
    def get_design_table(self, design_infer_labels):
        query = self.session.query(models.Design).filter(models.Design.design_infer_label.in_(design_infer_labels))
        return pd.read_sql(query.statement, self.session.bind, 'design_infer_label').to_dict()['design_pkey']

    # route_inference.py
    def get_design_and_tag(self, design_infer_labels):
        """
        -> design_pkey, design_infer_label, tag_value
        """
        # short_tall_labels = self.session.query(models.Design.design_infer_label)\
        #     .select_from(models.DesignTagLink)\
        #     .join(models.Design, models.Design.design_tag_pkey == models.DesignTagLink.design_tag_pkey)\
        #     .join(models.Tag, models.Tag.tag_pkey == models.DesignTagLink.tag_pkey)\
        #     .filter(models.Design.design_infer_label.in_(design_infer_labels))\
        #     .all()

        # short_tall_labels = list(map(lambda x: x[0], short_tall_labels))

        # labels = set(design_infer_labels + short_tall_labels)
        query = self.session.query(models.Design.design_pkey, models.Design.design_infer_label, models.Tag.tag_value)\
            .select_from(models.Design)\
            .outerjoin(models.DesignTagLink, models.Design.design_tag_pkey == models.DesignTagLink.design_tag_pkey)\
            .outerjoin(models.Tag, models.DesignTagLink.tag_pkey == models.Tag.tag_pkey)\
            .filter(models.Design.design_infer_label.in_(design_infer_labels))

        return pd.read_sql(query.statement, self.session.bind)

    # route_main.py, lc_inf.py store_id와 device_id로 device_pkey 가져오기
    def get_device_pkey(self, company_id, store_id, device_id):
        device_pkey = self.session.query(models.Device.device_pkey)\
            .select_from(models.Device)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(
                models.Company.company_id==company_id,
                models.Store.store_id==store_id,
                models.Device.device_id==device_id,
            ).first().device_pkey
        return device_pkey

    # lc_inf.py에서 각 로드셀의 무게값 가져오는 용도
    def get_loadcells_with_floor_by_device_pkey(self, device_pkey):
        query = self.session.query(models.Loadcell, models.Shelf.shelf_floor)\
            .join(models.Shelf)\
            .filter_by(device_pkey=device_pkey)\
            .order_by(models.Shelf.shelf_floor, models.Loadcell.loadcell_column)
        df = pd.read_sql(query.statement, self.session.bind)
        return df

    # lc_inf.py
    def get_cells_by_device_pkey(self, device_pkey):
        query = self.session.query(models.Cell)\
            .join(models.Shelf)\
            .filter_by(device_pkey=device_pkey)\
            .order_by(models.Shelf.shelf_floor, models.Cell.cell_column)
        df = pd.read_sql(query.statement, self.session.bind)
        return df

    # loadcell_snapshot.py
    def get_cell_column_length(self, company_id, store_id, device_id, shelf_floor):
        query = self.session.query(models.Cell)\
            .select_from(models.Cell)\
            .join(models.Shelf, models.Shelf.shelf_pkey==models.Cell.shelf_pkey)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(
                models.Company.company_id==company_id,
                models.Store.store_id==store_id,
                models.Device.device_id==device_id,
                models.Shelf.shelf_floor==shelf_floor
            ).all()
        return len(query)

    def get_cells_by_shelf_pkey(self, shelf_pkey):
        query = self.session.query(models.VaccineCell)\
            .filter(
                models.VaccineCell.shelf_pkey==shelf_pkey
            ).order_by(models.VaccineCell.cell_column)
        df = pd.read_sql(query.statement, self.session.bind)
        return df

    # lc_inf.py stock을 dict 형식으로 담아서 반환
    def get_stocks(self, cell_pkey):
        query_rst = self.session.query(models.Stock, models.Design.design_pkey)\
            .join(models.Design).filter(models.Stock.cell_pkey==cell_pkey).all()
        result = {}
        for rst in query_rst:
            result[rst[1]] = rst[0].stock_count
        return result

    # lc_inf.py design_pkey로 goods 정보 가져오기
    def get_goods_by_design_pkey(self, design_pkey):
        query_rst = self.session.query(models.Good.goods_id, models.Good.goods_name, models.Design.design_infer_label)\
            .select_from(models.Good)\
            .join(models.Design, models.Good.goods_id == models.Design.goods_id)\
            .filter_by(design_pkey=design_pkey).first()
        return query_rst

    # route_main.py
    def get_cell_pkey(self, company_id, store_id, device_id, shelf_floor, cell_column):
        result = self.session.query(models.Cell)\
            .select_from(models.Cell)\
            .join(models.Shelf, models.Shelf.shelf_pkey==models.Cell.shelf_pkey)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id,
                    models.Shelf.shelf_floor==shelf_floor,
                    models.Cell.cell_column==cell_column)\
            .first()
        return result.cell_pkey

    def get_model_name_from_shelf(self, device_pkey, shelf_floor):
        result = self.session.query(models.Model.model_name)\
            .select_from(models.Device, models.Shelf, models.Model)\
            .filter(
                models.Device.device_pkey == models.Shelf.device_pkey,
                models.Device.device_pkey == device_pkey,
                models.Shelf.shelf_floor == shelf_floor,
                models.Shelf.model_pkey == models.Model.model_pkey
            ).first()

        return result.model_name

    # route_main.py
    def get_design_pkey(self, goods_id=None, design_infer_label=None):
        design_pkey = None
        if goods_id:
            design_pkey = self.session.query(models.Design.design_pkey)\
                .filter_by(goods_id=goods_id)\
                .all()[-1].design_pkey
        if design_infer_label:
            design_pkey = self.session.query(models.Design.design_pkey)\
                .filter_by(design_infer_label=design_infer_label)\
                .first().design_pkey
        return design_pkey

    # lc_inf.py cell_pkey와 design_pkey에 해당되는 stocks가 있는지 체크
    def check_stocks(self, cell_pkey, design_pkey=None):
        if design_pkey is None:
            query = self.session.query(models.Stock).filter_by(cell_pkey=cell_pkey)
        else:
            query = self.session.query(models.Stock).filter_by(cell_pkey=cell_pkey, design_pkey=design_pkey)
        exists = query.first() is not None
        return exists

    # lc_inf.py cell_pkey와 design_pkey에 해당되는 stock 삭제
    def delete_stocks(self, cell_pkey, design_pkey=None):
        if design_pkey is None:
            self.session.query(models.Stock).filter_by(cell_pkey=cell_pkey).delete()
        else:
            self.session.query(models.Stock).filter_by(cell_pkey=cell_pkey, design_pkey=design_pkey).delete()
        # self.session.commit()

    # lc_inf.py cell_pkey와 design_pkey에 해당되는 stock 값을 바꾸기
    def update_stocks(self, cell_pkey, design_pkey, stock_count):
        self.session.query(models.Stock).filter_by(cell_pkey=cell_pkey, design_pkey=design_pkey).update({'stock_count': stock_count})
        # self.session.commit()

    # lc_inf.py cell_pkey와 design_pkey로 stock 추가
    def insert_stocks(self, cell_pkey, design_pkey, stock_count):
        stock = models.Stock(cell_pkey=cell_pkey ,design_pkey=design_pkey, stock_count=stock_count)
        self.session.add(stock)
        # self.session.commit()

    # lc_inf.py cell table에 있는 design_pkey_front를 업데이트
    def update_design_pkey_front(self, cell_pkey, design_pkey_front):
        query_rst = self.session.query(models.Cell).filter_by(cell_pkey=cell_pkey).update({'design_pkey_front': design_pkey_front})
        # self.session.commit()

    # change_counter.py
    def get_design_mean_n_std_weight(self, design_pkey):
        query_rst = self.session.query(models.Design).filter_by(design_pkey=design_pkey).first()
        return (query_rst.design_mean_weight, query_rst.design_std_weight)

    # route_main.py /admin_door_closed
    def update_cell_design_pkey_front(self, company_id, store_id, device_id, shelf_floor, infers):
        results = self.session.query(models.Cell)\
            .select_from(models.Cell)\
            .join(models.Shelf, models.Shelf.shelf_pkey==models.Cell.shelf_pkey)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id,
                    models.Shelf.shelf_floor==shelf_floor)\
            .order_by(models.Company.company_id,
                      models.Store.store_id,
                      models.Device.device_id,
                      models.Shelf.shelf_floor,
                      models.Cell.cell_column)\
            .all()

        for rst, infer in zip(results, infers):
            rst.design_pkey_front = infer if infer != -1 else None
        self.session.commit()

    # get cameras locations
    def get_cameras_locations(self, company_id=None, store_id=None, device_id=None, floor=None):
        """
        SELECT CAM.LOCATION
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CAMERAS CAM
        WHERE COM.COMPANY_ID = '{company_id}'
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = '{store_id}'
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = '{device_id}'
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.SHELF_FLOOR = {floor}
        AND SH.SHELF_PKEY = CAM.SHELF_PKEY
        AND CAM.USE_FLAG = TRUE
        """
        query = self.session.query(models.Cameras.location)\
            .select_from(models.Company, models.Store, models.Device, models.Shelf)\
            .filter(
                models.Company.company_id == company_id,
                models.Company.company_pkey == models.Store.company_pkey,
                models.Store.store_id == store_id,
                models.Store.store_pkey == models.Device.store_pkey,
                models.Device.device_id == device_id,
                models.Device.device_pkey == models.Shelf.device_pkey,
                models.Shelf.shelf_floor == floor,
                models.Shelf.shelf_pkey == models.Cameras.shelf_pkey,
                models.Cameras.use_flag == True
            )
        df = pd.read_sql(query.statement, self.session.bind)

        return df

    # gui_json.py
    def get_device_operation(self, company_id=None, store_id=None):
        query = self.session.query(models.Device)\
            .select_from(models.Device)\
            .filter_by(operation=True)\
            .order_by(models.Device.device_id)
        if store_id is not None:
            query = query.join(models.Store, models.Store.store_pkey==models.Device.store_pkey)
            if company_id is not None:
                query = query.join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
                    .filter(models.Company.company_id==company_id)
            query = query.filter(models.Store.store_id==store_id)

        df = pd.read_sql(query.statement, self.session.bind)
        return df

    def get_device_alarm(self, device_pkey):
        query = self.session.query(models.Device).filter_by(device_pkey = device_pkey)
        df = pd.read_sql(query.statement, self.session.bind)
        return df.alarm

    # route_main.py
    def get_goods_name_cell_pkey(self, device_pkey, floor, cell):
        result = self.session.query(models.Cell.cell_pkey, models.Good.goods_name) \
                    .join(models.Shelf, models.Cell.shelf_pkey == models.Shelf.shelf_pkey)\
                    .filter(models.Shelf.shelf_floor == floor).filter(models.Cell.cell_column == cell)\
                    .join(models.Device, models.Shelf.device_pkey == models.Device.device_pkey)\
                    .filter_by(device_pkey = device_pkey)\
                    .join(models.Design, models.Cell.design_pkey_master == models.Design.design_pkey)\
                    .join(models.Good, models.Good.goods_id == models.Design.goods_id)
        df = pd.read_sql(result.statement, self.session.bind)
        return df.goods_name

    # route_main.py
    def get_max_floor(self, device_pkey):
        result = self.session.query(models.Shelf).join(models.Device).filter_by(device_pkey = device_pkey)
        df = pd.read_sql(result.statement, self.session.bind)
        return max(df.shelf_floor)

    # gui_json.py
    def get_company_name(self, company_id):
        result = self.session.query(models.Company).filter_by(company_id=company_id)
        df = pd.read_sql(result.statement, self.session.bind)
        return df.company_name

    # gui_json.py
    def get_store_name(self, company_id, store_id):
        result = self.session.query(models.Store)\
            .select_from(models.Store)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(models.Company.company_id==company_id, models.Store.store_id==store_id)
        df = pd.read_sql(result.statement, self.session.bind)
        return df.store_name

    # /update_goods
    def update_goods(self, goods_id, goods_name, design_mean_weight):
        query = self.session.query(models.Good, models.Design)\
            .select_from(models.Good)\
            .join(models.Design, models.Good.goods_id == models.Design.goods_id)\
            .filter_by(goods_id=goods_id)
        result = query.first()
        result[0].goods_name = goods_name
        result[1].design_mean_weight = design_mean_weight
        self.session.commit()
        # return pd.read_sql(query.statement, session.bind)

    # /update_pog
    def update_pog(self, cell_pkey, design_pkey):
        result = self.session.query(models.Cell).filter_by(cell_pkey=cell_pkey).first()
        result.design_pkey_master = design_pkey
        result.design_pkey_front = design_pkey
        self.session.commit()


    # /regist_goods
    def insert_goods(self, goods_id, goods_name, design_infer_label, design_mean_weight, design_std_weight=None):
        goods = models.Good(goods_id=goods_id, goods_name=goods_name)
        self.session.add(goods)
        designs = models.Design(goods_id=goods_id,
                                design_mean_weight=design_mean_weight,
                                design_std_weight=design_std_weight if design_std_weight is not None else design_mean_weight * 0.1,
                                design_infer_label=design_infer_label)
        self.session.add(designs)
        self.session.commit()


    def get_designs_by_design_pkey(self, design_pkey):
        return self.session.query(models.Design).filter_by(design_pkey=design_pkey).first()


    def insert_trade_log(
        self,
        trade_log_no,
        trade_log_date,
        trade_log_time,
        company_id,
        store_id,
        device_id,
        shelf_floor,
        cell_column,
        goods_id,
        goods_name,
        goods_label,
        goods_count,
        stock_left,
        goods_mean_weight,
        goods_std_weight,
        open_weight,
        close_weight,
        duration,
        work_user,
        work_type,
        sale_price,
        status_code
    ):
        i = insert(self.trade_log)
        i = i.values({
            "trade_log_no": trade_log_no,
            "trade_log_date": trade_log_date,
            "trade_log_time": trade_log_time,
            "company_id": company_id,
            "store_id": store_id,
            "device_id": device_id,
            "shelf_floor": shelf_floor,
            "cell_column": cell_column,
            "goods_id": goods_id,
            "goods_name": goods_name,
            "goods_label": goods_label,
            "goods_count": goods_count,
            "stock_left": stock_left,
            "goods_mean_weight": goods_mean_weight,
            "goods_std_weight": goods_std_weight,
            "open_weight": open_weight,
            "close_weight": close_weight,
            "duration": duration,
            "work_user": work_user,
            "work_type": work_type,
            "sale_price": sale_price,
            "status_code": status_code})
        self.session.execute(i)
        # self.session.commit()

    # route_main.py
    def get_last_trade_log_no(self):
        result = self.session.query(models.TradeLog.trade_log_no).order_by(models.TradeLog.trade_log_pkey.desc()).first()
        if result:
            return result.trade_log_no
        else:
            return 0

    # route_main.py
    def get_last_trade(self, company_id, store_id, device_id):
        result = self.session.query(models.Trade)\
            .select_from(models.Trade)\
            .join(models.Device, models.Device.device_pkey==models.Trade.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id)\
            .order_by(models.Trade.trade_pkey.desc()).first()
        if result is None:
            return self.insert_trade(company_id, store_id, device_id)
        else:
            return result.trade_pkey, result.trade_date, result.trade_time, result.device_pkey
            
    # route_main.py
    def insert_trade(self, company_id, store_id, device_id):
        device_pkey = self.get_device_pkey(company_id, store_id, device_id)
        trade = models.Trade(device_pkey=device_pkey)
        self.session.add(trade)
        self.session.commit()
        return trade.trade_pkey, trade.trade_date, trade.trade_time, trade.device_pkey

    def get_cigar_stocks_by_csd_id(self, company_id, store_id, device_id):
        """
        SELECT SH.SHELF_FLOOR, CS.CELL_COLUMN, GOO.GOODS_ID, GOO.GOODS_NAME, CS.TOTAL_CNT, CS.STOCK_COUNT_LOW_ALERT_LIMIT
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CIGAR_CELLS CS, GOODS GOO, DESIGNS DES
        WHERE COM.COMPANY_ID = '{company_id}'
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = '{store_id}'
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = '{device_id}'
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.SHELF_PKEY = CS.SHELF_PKEY
        AND CS.DESIGN_PKEY_MASTER = DES.DESIGN_PKEY
        AND DES.GOODS_ID = GOO.GOODS_ID
        AND CS.STOCK_COUNT_LOW_ALERT_LIMIT IS NOT NULL
        AND CS.TOTAL_CNT <= CS.STOCK_COUNT_LOW_ALERT_LIMIT
        """
        result = self.session.query(models.Shelf.shelf_floor,
                                    models.CigarCell.cell_column,
                                    models.Good.goods_id,
                                    models.Good.goods_name,
                                    models.CigarCell.total_cnt,
                                    models.CigarCell.stock_count_low_alert_limit)\
            .select_from(models.Company, models.Store, models.Device, models.Shelf, models.CigarCell, models.Good, models.Design)\
            .filter(
                models.Company.company_id == company_id,
                models.Company.company_pkey == models.Store.company_pkey,
                models.Store.store_id == store_id,
                models.Store.store_pkey == models.Device.store_pkey,
                models.Device.device_id == device_id,
                models.Device.device_pkey == models.Shelf.device_pkey,
                models.Shelf.shelf_pkey == models.CigarCell.shelf_pkey,
                models.CigarCell.design_pkey_master == models.Design.design_pkey,
                models.Design.goods_id == models.Good.goods_id,
                models.CigarCell.stock_count_low_alert_limit.isnot(None),
                models.CigarCell.total_cnt < models.CigarCell.stock_count_low_alert_limit
            ).order_by(models.Shelf.shelf_floor, models.CigarCell.cell_column).all()

        return result

    def get_stocks_by_csd_id(self, company_id, store_id, device_id):
        result = self.session.query(models.Shelf.shelf_floor,
                                    models.Cell.cell_column,
                                    models.Good.goods_id,
                                    models.Good.goods_name,
                                    models.Stock.stock_count,
                                    models.Cell.stock_count_low_alert_limit)\
            .select_from(models.Stock)\
            .join(models.Cell, models.Cell.cell_pkey==models.Stock.cell_pkey)\
            .join(models.Shelf, models.Shelf.shelf_pkey==models.Cell.shelf_pkey)\
            .join(models.Device, models.Device.device_pkey==models.Shelf.device_pkey)\
            .join(models.Store, models.Store.store_pkey==models.Device.store_pkey)\
            .join(models.Company, models.Company.company_pkey==models.Store.company_pkey)\
            .join(models.Design, models.Design.design_pkey==models.Stock.design_pkey)\
            .join(models.Good, models.Good.goods_id==models.Design.goods_id)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id,
                    models.Stock.design_pkey==models.Cell.design_pkey_master,
                    models.Cell.load_cell_mode=='abs',
                    models.Cell.stock_count_low_alert_limit.isnot(None),
                    models.Stock.stock_count<=models.Cell.stock_count_low_alert_limit)\
            .all()
        return result


    def get_sale_price(self, store_id, design_pkey):
        sale_price = self.session.query(models.Sale.sale_price)\
            .select_from(models.Sale)\
            .join(models.Store, models.Store.store_pkey==models.Sale.store_pkey)\
            .filter(models.Sale.design_pkey==design_pkey,
                    models.Store.store_id==store_id)\
            .order_by(models.Sale.sale_reg_date.desc())\
            .first()
        return None if sale_price is None else sale_price[0]

    # Model 별 Tag 정보 select
    def get_tag_info(self, design_infer_label, model_pkey):
        """
        SELECT DES.DESIGN_PKEY, DES.DESIGN_INFER_LABEL, T.TAG_VALUE
        FROM DESIGNS DES, DESIGN_TAG_LINK DTL, TAG T
        WHERE DES.DESIGN_INFER_LABEL = {DESIGN_INFER_LABEL}
        AND DES.DESIGN_TAG_PKEY = DTL.DESIGN_TAG_PKEY
        AND DTL.TAG_PKEY = T.TAG_PKEY
        AND DTL.MODEL_PKEY = {MODEL_PKEY}
        LIMIT 1
        """
        tag_info = self.session.query(models.Tag.tag_value)\
            .select_from(models.Design, models.DesignTagLink, models.Tag)\
            .filter(
                models.Design.design_infer_label == design_infer_label,
                models.Design.design_tag_pkey == models.DesignTagLink.design_tag_pkey,
                models.DesignTagLink.tag_pkey == models.Tag.tag_pkey,
                models.DesignTagLink.model_pkey == model_pkey
            ).first()

        return None if tag_info is None else tag_info[0]

    # Goods Id로 Goods Name get.
    def get_goods_name(self, goods_id):
        """
        SELECT GOODS_NAME
        FROM GOODS
        WHERE GOODS_ID = {goods_id}
        """
        goods_name = self.session.query(models.Good.goods_name)\
            .select_from(models.Good)\
            .filter(
                models.Good.goods_id == goods_id
            ).first()

        return None if goods_name is None else goods_name[0]

    # Goods Id로 design_label get.
    def get_design_label(self, goods_id, design_label_list):
        design_infer_label = self.session.query(models.Design.design_infer_label)\
            .select_from(models.Design)\
            .filter(
                models.Design.goods_id == goods_id
            ).all()
        for design_idx in design_infer_label:
            if design_idx[0] in design_label_list:
                return design_idx[0]
        return None

    # cigar_cells 테이블의 design_pkey_master 정보 get.
    def get_cell_and_design_keys(self, company_id, store_id, device_id, shelf_floor, cell_column):
        """
        SELECT CS.CELL_PKEY, CS.DESIGN_PKEY_MASTER
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CIGAR_CELLS CS
        WHERE COM.COMPANY_ID = '{company_id}'
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = '{store_id}'
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = '{device_id}'
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.SHELF_FLOOR = {shelf_floor}
        AND SH.SHELF_PKEY = CS.SHELF_PKEY
        AND CS.CELL_COLUMN = {cell_column}
        """
        cell_design_keys_dict = {'cell_pkey': None, 'design_pkey_master': None}

        query = self.session.query(models.CigarCell.cell_pkey, 
                                   models.CigarCell.design_pkey_master,
                                   models.CigarCell.stock_count_max)\
            .select_from(models.Company, models.Store, models.Device, models.Shelf, models.CigarCell) \
            .filter(
                models.Company.company_id == company_id,
                models.Company.company_pkey == models.Store.company_pkey,
                models.Store.store_id == store_id,
                models.Store.store_pkey == models.Device.store_pkey,
                models.Device.device_id == device_id,
                models.Device.device_pkey == models.Shelf.device_pkey,
                models.Shelf.shelf_floor == shelf_floor,
                models.Shelf.shelf_pkey == models.CigarCell.shelf_pkey,
                models.CigarCell.cell_column == cell_column
            ).first()

        if query is not None:
            cell_design_keys_dict['cell_pkey'] = query[0]
            cell_design_keys_dict['design_pkey_master'] = query[1]
            cell_design_keys_dict['stock_count_max'] = query[2]

        return cell_design_keys_dict

    # companyId, storeId, deviceId로 모든 칸의 POG 상품 정보 가져오기
    def get_pog_goods(self, company_id, store_id, device_id):
        """
        select shelves.shelf_floor, cells.cell_column, goods.goods_name, goods.goods_id
        from companies
        left join stores on companies.company_pkey = stores.company_pkey
        left join devices on stores.store_pkey = devices.store_pkey
        left join shelves on shelves.device_pkey = devices.device_pkey
        left join cells on shelves.shelf_pkey = cells.shelf_pkey
        left join designs on  designs.design_pkey = cells.design_pkey_master
        left join goods on goods.goods_id = designs.goods_id
        where companies.company_id = {company_id} and stores.store_id = {store_id} and devices.device_id = {device_id}
        order by shelves.shelf_floor, cells.cell_column
        """
        result = \
            self.session.query(models.Shelf.shelf_floor, models.Cell.cell_column, \
                                models.Good.goods_name, models.Good.goods_id)\
            .select_from(models.Company)\
            .join(models.Store, models.Store.company_pkey==models.Company.company_pkey)\
            .join(models.Device, models.Device.store_pkey==models.Store.store_pkey)\
            .join(models.Shelf, models.Shelf.device_pkey==models.Device.device_pkey)\
            .join(models.Cell, models.Cell.shelf_pkey==models.Shelf.shelf_pkey)\
            .join(models.Design, models.Cell.design_pkey_master == models.Design.design_pkey)\
            .join(models.Good, models.Good.goods_id == models.Design.goods_id)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id)\
            .order_by(models.Shelf.shelf_floor,
                    models.Cell.cell_column)
        df = pd.read_sql(result.statement, self.session.bind)
        return df

    # Log handler 를 지정 하기 위한 device 스토리지 타입 정보 get
    def get_device_storage_type(self, store_id, device_id):
        """
        SELECT DEVICE_STORAGE_TYPE
        FROM STORES ST, DEVICES DEV
        WHERE  DEV.DEVICE_ID = {device_id}
        AND DEV.STORE_PKEY = ST.STORE_PKEY
        AND ST.STORE_ID = {store_id}
        """
        dev_storage_type = self.session.query(models.Device.device_storage_type) \
            .select_from(models.Device, models.Store) \
            .filter(
                models.Device.device_id == device_id,
                models.Device.store_pkey == models.Store.store_pkey,
                models.Store.store_id == store_id
            ).first()[0]

        return 'CD' if dev_storage_type is None else dev_storage_type

    # Cigar Cells Update Query
    def update_cigar_cells(self, cigar_cells_dict):
        """
        UPDATE CIGAR_CELLS
        SET DESIGN_PKEY_MASTER = {cigar_cells_dict['d_p_m']},
        DESIGN_PKEY_FIRST = {cigar_cells_dict['d_p_f']},
        DESIGN_PKEY_SECOND = {cigar_cells_dict['d_p_s']},
        DESIGN_PKEY_THIRD = {cigar_cells_dict['d_p_t']},
        TOTAL_CNT = {cigar_cells_dict['total_cnt']}
        WHERE CELL_PKEY = {cigar_cells_dict['cell_pkey']}
        """
        result = self.session.query(models.CigarCell).filter_by(cell_pkey=cigar_cells_dict['cell_pkey']).first()
        result.design_pkey_master = cigar_cells_dict['d_p_m']
        result.design_pkey_first = cigar_cells_dict['d_p_f']
        result.design_pkey_second = cigar_cells_dict['d_p_s']
        result.design_pkey_third = cigar_cells_dict['d_p_t']
        result.total_cnt = cigar_cells_dict['total_cnt']
        self.session.commit()

    # Delete Cigar Stocks
    def delete_cigar_stocks(self, cell_pkey):
        """
        DELETE FROM CIGAR_STOCKS WHERE CELL_PKEY = {cell_pkey}
        """
        self.session.query(models.CigarStock).filter_by(cell_pkey=cell_pkey).delete()

    # 담배 companyId, storeId, deviceId로 모든 칸의 POG 상품 정보 가져오기
    def get_ciga_pog_goods(self, company_id, store_id, device_id):
        """
        select shelves.shelf_floor, cigar_cells.cell_column, goods.goods_name, goods.goods_id, designs.design_infer_label
        from companies
        left join stores on companies.company_pkey = stores.company_pkey
        left join devices on stores.store_pkey = devices.store_pkey
        left join shelves on shelves.device_pkey = devices.device_pkey
        left join cigar_cells on shelves.shelf_pkey = cigar_cells.shelf_pkey
        left join designs on  designs.design_pkey = cigar_cells.design_pkey_master
        left join goods on goods.goods_id = designs.goods_id
        where companies.company_id = {company_id} and stores.store_id = {store_id} and devices.device_id = {device_id}
        order by shelves.shelf_floor, cigar_cells.cell_column
        """
        result = \
            self.session.query(models.Shelf.shelf_floor, models.CigarCell.cell_column, \
                                models.Good.goods_name, models.Good.goods_id, models.Design.design_infer_label)\
            .select_from(models.Company)\
            .join(models.Store, models.Store.company_pkey==models.Company.company_pkey)\
            .join(models.Device, models.Device.store_pkey==models.Store.store_pkey)\
            .join(models.Shelf, models.Shelf.device_pkey==models.Device.device_pkey)\
            .join(models.CigarCell, models.CigarCell.shelf_pkey==models.Shelf.shelf_pkey)\
            .join(models.Design, models.CigarCell.design_pkey_master == models.Design.design_pkey)\
            .join(models.Good, models.Good.goods_id == models.Design.goods_id)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id)\
            .order_by(models.Shelf.shelf_floor,
                    models.CigarCell.cell_column)
        df = pd.read_sql(result.statement, self.session.bind)
        return df

    # device_type 가져오기
    def get_device_type(self, company_id , store_id, device_id):
        """
        select devices.device_storage_type
        from companies
        left join stores on companies.company_pkey = stores.company_pkey
        left join devices on stores.store_pkey = devices.store_pkey
        where companies.company_id = '0001' and stores.store_id = '00888' and devices.device_id = 's_00008'
        """
        d_storage_type = \
            self.session.query(models.Device.device_storage_type)\
            .select_from(models.Company)\
            .join(models.Store, models.Store.company_pkey==models.Company.company_pkey)\
            .join(models.Device, models.Device.store_pkey==models.Store.store_pkey)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id)\
            .first()

        return None if d_storage_type is None else d_storage_type[0]

    # Insert Cigar Stocks
    def insert_cigar_stocks(self, cell_pkey, design_pkey, stock_count):
        """
        INSERT INTO CIGAR_STOCKS (CELL_PKEY, DESIGN_PKEY, STOCK_COUNT)
        VALUES ({cell_pkey}, {design_pkey}, {stock_count})
        """
        stock = models.CigarStock(cell_pkey=cell_pkey, design_pkey=design_pkey, stock_count=stock_count)
        self.session.add(stock)

    # Select Master Designs Pkey
    def get_designs_pkey_master(self, company_id, store_id, device_id):
        """
        SELECT CEL.CELL_PKEY, SH.SHELF_FLOOR, CEL.CELL_COLUMN,
        CEL.DESIGN_PKEY_MASTER, CEL.INFERENCE_MODE, DES.DESIGN_INFER_LABEL
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CELLS CEL, DESIGNS DES
        WHERE COM.COMPANY_ID = '{company_id}'
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = '{store_id}'
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = '{device_id}'
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.SHELF_PKEY = CEL.SHELF_PKEY
        AND CEL.DESIGN_PKEY_MASTER = DES.DESIGN_PKEY
        ORDER BY SH.SHELF_FLOOR, CEL.CELL_COLUMN
        """
        result = self.session.query \
            (models.Cell.cell_pkey, models.Device.device_pkey, models.Shelf.shelf_floor, models.Cell.cell_column,
             models.Cell.design_pkey_front, models.Cell.design_pkey_master, models.Cell.inference_mode, models.Design.design_infer_label) \
            .select_from(models.Company, models.Store, models.Device, models.Shelf, models.Cell, models.Design) \
            .filter(
            models.Company.company_id == company_id,
            models.Company.company_pkey == models.Store.company_pkey,
            models.Store.store_id == store_id,
            models.Store.store_pkey == models.Device.store_pkey,
            models.Device.device_id == device_id,
            models.Device.device_pkey == models.Shelf.device_pkey,
            models.Shelf.shelf_pkey == models.Cell.shelf_pkey,
            models.Cell.design_pkey_master == models.Design.design_pkey
        ).order_by(models.Shelf.shelf_floor, models.Cell.cell_column)

        df = pd.read_sql(result.statement, self.session.bind)

        """
        cell_pkey, device_pkey, shelf_floor, cell_column, design_pkey_front, design_pkey_master, inference_mode, design_infer_label
        (740,        1,             0,           0,             301,                301,              'mix',     'beat_coffee_p_5p')
        (741,        2,             0,           1,             323,                323,              'lc',      'alive_orange')
        (742,        3,             0,           2,             350,                350,              'mix',     'lemonaid_original_pet_330')
        ............
        """
        cells_master_dict = {}
        for idx, (
        cell_pkey, device_pkey, shelf_floor, cell_column, design_pkey_front, design_pkey_master, inference_mode, design_infer_label) in enumerate(
                zip(df['cell_pkey'].values, df['device_pkey'].values, df['shelf_floor'].values, df['cell_column'].values,
                    df['design_pkey_front'].values, df['design_pkey_master'].values, df['inference_mode'].values, df['design_infer_label'].values)):
            cells_master_dict[idx] = {'cell_pkey': cell_pkey,
                                      'device_pkey': device_pkey,
                                      'shelf_floor': shelf_floor,
                                      'cell_column': cell_column,
                                      'design_pkey_front': design_pkey_front,
                                      'design_pkey_master': design_pkey_master,
                                      'inference_mode': inference_mode,
                                      'design_infer_label': design_infer_label}

        return cells_master_dict

    # Select 강제 'lc' mode goods list
    def get_lc_mode_goods_list(self, design_pkey, model_name):
        """
        SELECT COUNT(*)
        FROM LC_MODE_GOODS_LIST LC, DESIGNS DES
        WHERE LC.GOODS_ID = DES.GOODS_ID
        AND DES.DESIGN_PKEY = {design_pkey}
		AND LC.MODEL_NAME = {model_name}
        """
        result = self.session.query \
            (models.LcModeGoodsList.goods_id) \
            .select_from(models.LcModeGoodsList, models.Design) \
            .filter(
                models.LcModeGoodsList.goods_id == models.Design.goods_id,
                models.Design.design_pkey == design_pkey,
                models.LcModeGoodsList.model_name == model_name
            ).first()

        return 1 if result else 0

# Select Master Designs Pkey
    def get_designs_pkey_master_cigar(self, company_id, store_id, device_id, floor):
        """
        SELECT CEL.CELL_PKEY, SH.SHELF_FLOOR, CEL.CELL_COLUMN, CEL.INFERENCE_MODE,
        CEL.DESIGN_PKEY_MASTER, CEL.INFERENCE_MODE, DES.DESIGN_INFER_LABEL
        FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CELLS CEL, DESIGNS DES
        WHERE COM.COMPANY_ID = '{company_id}'
        AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
        AND ST.STORE_ID = '{store_id}'
        AND ST.STORE_PKEY = DEV.STORE_PKEY
        AND DEV.DEVICE_ID = '{device_id}'
        AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
        AND SH.SHELF_PKEY = CEL.SHELF_PKEY
        AND CEL.DESIGN_PKEY_MASTER = DES.DESIGN_PKEY
        ORDER BY SH.SHELF_FLOOR, CEL.CELL_COLUMN
        """
        result = self.session.query \
            (models.CigarCell.cell_pkey, models.Device.device_pkey, models.Shelf.shelf_floor, models.CigarCell.cell_column,
             models.CigarCell.design_pkey_first, models.CigarCell.design_pkey_second, models.CigarCell.design_pkey_third, models.CigarCell.design_pkey_master, models.CigarCell.inference_mode, models.Design.design_infer_label) \
            .select_from(models.Company, models.Store, models.Device, models.Shelf, models.CigarCell, models.Design) \
            .filter(
            models.Company.company_id == company_id,
            models.Company.company_pkey == models.Store.company_pkey,
            models.Store.store_id == store_id,
            models.Store.store_pkey == models.Device.store_pkey,
            models.Device.device_id == device_id,
            models.Device.device_pkey == models.Shelf.device_pkey,
            models.Shelf.shelf_floor == floor,
            models.Shelf.shelf_pkey == models.CigarCell.shelf_pkey,
            models.CigarCell.design_pkey_master == models.Design.design_pkey
        ).order_by(models.CigarCell.cell_column)

        df = pd.read_sql(result.statement, self.session.bind)

        """
        cell_pkey, device_pkey, shelf_floor, cell_column, design_pkey_front, design_pkey_master, inference_mode, design_infer_label
        (740,        1,             0,           0,             301,                301,              'mix',     'beat_coffee_p_5p')
        (741,        2,             0,           1,             323,                323,              'lc',      'alive_orange')
        (742,        3,             0,           2,             350,                350,              'mix',     'lemonaid_original_pet_330')
        ............
        """
        cells_master_dict = {}
        for idx, (
        cell_pkey, device_pkey, shelf_floor, cell_column, design_pkey_first, design_pkey_second, design_pkey_third, design_pkey_master, inference_mode, design_infer_label) in enumerate(
                zip(df['cell_pkey'].values, df['device_pkey'].values, df['shelf_floor'].values, df['cell_column'].values,
                    df['design_pkey_first'].values, df['design_pkey_second'].values, df['design_pkey_third'].values, df['design_pkey_master'].values, df['inference_mode'].values, df['design_infer_label'].values)):
            cells_master_dict[idx] = {'cell_pkey': cell_pkey,
                                      'device_pkey': device_pkey,
                                      'shelf_floor': shelf_floor,
                                      'cell_column': cell_column,
                                      'design_pkey_first': None if pd.isna(design_pkey_first) else design_pkey_first,
                                      'design_pkey_second': None if pd.isna(design_pkey_second) else design_pkey_second,
                                      'design_pkey_third': None if pd.isna(design_pkey_third) else design_pkey_third,
                                      'design_pkey_master': design_pkey_master,
                                      'inference_mode': inference_mode,
                                      'design_infer_label': design_infer_label}

        return cells_master_dict


    # Select Cigar Stocks
    def get_cigar_stocks(self, cell_pkey):
        """
        SELECT DESIGN_PKEY, STOCK_COUNT
        FROM CIGAR_STOCKS
        WHERE CELL_PKEY = {cell_pkey}
        """
        result = self.session.query(models.CigarStock).filter_by(cell_pkey=cell_pkey)
        df = pd.read_sql(result.statement, self.session.bind)

        temp_dict = {}
        for data in zip(df['design_pkey'].values, df['stock_count'].values):
            temp_dict[data[0]] = data[1]

        return {} if len(temp_dict) <= 0 else temp_dict

    def get_device_master_pkey(self, company_id, store_id, device_id):
        """
                SELECT CEL.DESIGN_PKEY_MASTER
                FROM COMPANIES COM, STORES ST, DEVICES DEV, SHELVES SH, CELLS CEL, DESIGNS DES
                WHERE COM.COMPANY_ID = '{company_id}'
                AND COM.COMPANY_PKEY = ST.COMPANY_PKEY
                AND ST.STORE_ID = '{store_id}'
                AND ST.STORE_PKEY = DEV.STORE_PKEY
                AND DEV.DEVICE_ID = '{device_id}'
                AND DEV.DEVICE_PKEY = SH.DEVICE_PKEY
                AND SH.SHELF_PKEY = CEL.SHELF_PKEY
                AND CEL.DESIGN_PKEY_MASTER = DES.DESIGN_PKEY
                GROUP BY CEL.DESIGN_PKEY_MASTER
        """
        result = self.session.query \
            (models.Cell.design_pkey_master, models.Design.design_infer_label) \
            .filter(
            models.Company.company_id == company_id,
            models.Company.company_pkey == models.Store.company_pkey,
            models.Store.store_id == store_id,
            models.Store.store_pkey == models.Device.store_pkey,
            models.Device.device_id == device_id,
            models.Device.device_pkey == models.Shelf.device_pkey,
            models.Shelf.shelf_pkey == models.Cell.shelf_pkey,
            models.Cell.design_pkey_master == models.Design.design_pkey
        )
        """
        design_pkey_master, design_infer_label
        666                 cass_zero_355
        674                 belgium_weizen_500
        ...............
        """
        return pd.read_sql(result.statement, self.session.bind, 'design_pkey_master').to_dict()['design_infer_label']
    
    # Select (cell_column, shelf_floor) cigar_cells, shelves
    def get_cell_column_shelf_floor(self, cell_pkey):
        """
        SELECT CS.CELL_COLUMN, SH.SHELF_FLOOR
        FROM CIGAR_CELLS CS, SHELVES SH
        WHERE CS.CELL_PKEY = {cell_pkey}
        AND CS.SHELF_PKEY = SH.SHELF_PKEY
        """
        data = self.session.query(models.CigarCell.cell_column, models.Shelf.shelf_floor) \
            .select_from(models.CigarCell, models.Shelf) \
            .filter(
                models.CigarCell.cell_pkey == cell_pkey,
                models.CigarCell.shelf_pkey == models.Shelf.shelf_pkey
            ).first()

        return None if len(data) <= 0 else {'cell_column': data[0], 'shelf_floor': data[1]}

    # Insert Cigar Trade Log
    def insert_cigar_trade_log(
        self,
        cigar_trade_log_no,
        cigar_trade_log_date,
        cigar_trade_log_time,
        company_id,
        store_id,
        device_id,
        shelf_floor,
        cell_column,
        goods_id,
        goods_name,
        goods_label,
        goods_count,
        stock_left,
        duration,
        work_user,
        work_type,
        status_code,
        sale_price,
        total_sale_price
    ):
        i = insert(self.cigar_trade_log)
        i = i.values({
            "cigar_trade_log_no": cigar_trade_log_no,
            "cigar_trade_log_date": cigar_trade_log_date,
            "cigar_trade_log_time": cigar_trade_log_time,
            "company_id": company_id,
            "store_id": store_id,
            "device_id": device_id,
            "shelf_floor": shelf_floor,
            "cell_column": cell_column,
            "goods_id": goods_id,
            "goods_name": goods_name,
            "goods_label": goods_label,
            "goods_count": goods_count,
            "stock_left": stock_left,
            "duration": duration,
            "work_user": work_user,
            "work_type": work_type,
            "status_code": status_code,
            "sale_price": sale_price,
            "total_sale_price": total_sale_price})
        self.session.execute(i)

    # Select (cell_column, shelf_floor) vaccine_cells, shelves
    def get_cell_column_shelf_floor_vaccine(self, cell_pkey):
        """
        SELECT CS.CELL_COLUMN, SH.SHELF_FLOOR
        FROM VACCINE_CELLS CS, SHELVES SH
        WHERE CS.CELL_PKEY = {cell_pkey}
        AND CS.SHELF_PKEY = SH.SHELF_PKEY
        """
        data = self.session.query(models.VaccineCell.cell_column, models.Shelf.shelf_floor) \
            .select_from(models.VaccineCell, models.Shelf) \
            .filter(
                models.VaccineCell.cell_pkey == cell_pkey,
                models.VaccineCell.shelf_pkey == models.Shelf.shelf_pkey
            ).first()

        return None if len(data) <= 0 else {'cell_column': data[0], 'shelf_floor': data[1]}


    # Vaccine Cells Update Query
    def update_vaccine_cells(self, vaccine_cells_dict):
        """
        UPDATE VACCINE_CELLS
        SET TOTAL_CNT = {vaccine_cells_dict['total_cnt']}
        WHERE CELL_PKEY = {vaccine_cells_dict['cell_pkey']}
        """
        result = self.session.query(models.VaccineCell).filter_by(cell_pkey=vaccine_cells_dict['cell_pkey']).first()
        result.total_cnt = vaccine_cells_dict['total_cnt']
        self.session.commit()

    # 백신 결제 내역 검수 테이블 master insert
    def insert_vaccine_trade_check(
        self,
        vaccine_trade_date,
        vaccine_trade_time,
        company_id,
        store_id,
        device_id,
        total_sale_price,
        qr_data,
        user_level
    ):
        i = insert(self.vaccine_trade_check)
        i = i.values({
            "vaccine_trade_date": vaccine_trade_date,
            "vaccine_trade_time": vaccine_trade_time,
            "company_id": company_id,
            "store_id": store_id,
            "device_id": device_id,
            "total_sale_price": total_sale_price,
            "qr_data": qr_data,
            "user_level":user_level})
        self.session.execute(i)
    
    # 백신 결제 내역 검수 테이블 sub insert
    def insert_vaccine_trade_pog(
        self,
        vaccine_trade_check_pkey,
        vaccine_trade_no,
        shelf_floor,
        cell_column,
        goods_id,
        goods_name,
        goods_label,
        goods_count,
        sale_price
    ):
        i = insert(self.vaccine_trade_pog)
        i = i.values({
            "vaccine_trade_check_pkey": vaccine_trade_check_pkey,
            "vaccine_trade_no": vaccine_trade_no,
            "shelf_floor": shelf_floor,
            "cell_column": cell_column,
            "goods_id": goods_id,
            "goods_name": goods_name,
            "goods_label": goods_label,
            "goods_count": goods_count,
            "sale_price": sale_price})
        self.session.execute(i)

    def get_vaccine_trade_check_pkey(
        self, 
        vaccine_trade_date,
        vaccine_trade_time,
        company_id,
        store_id,
        device_id,
    ):
        trade_check_pkey = self.session.query(models.VaccineTradeCheck.vaccine_trade_check_pkey) \
            .select_from(models.VaccineTradeCheck) \
            .filter(
                models.VaccineTradeCheck.company_id == company_id,
                models.VaccineTradeCheck.store_id == store_id,
                models.VaccineTradeCheck.device_id == device_id,
                models.VaccineTradeCheck.vaccine_trade_date == vaccine_trade_date,
                models.VaccineTradeCheck.vaccine_trade_time == vaccine_trade_time
            ).first()[0]
        return None if trade_check_pkey is None else trade_check_pkey
    # Select Vaccine Stocks
    def get_vaccine_stocks(self, cell_pkey):
        """
        SELECT DESIGN_PKEY, STOCK_COUNT
        FROM VACCINE_STOCKS
        WHERE CELL_PKEY = {cell_pkey}
        """
        result = self.session.query(models.VaccineStock).filter_by(cell_pkey=cell_pkey)
        df = pd.read_sql(result.statement, self.session.bind)

        temp_dict = {}
        for data in zip(df['design_pkey'].values, df['stock_count'].values):
            temp_dict[data[0]] = data[1]

        return {} if len(temp_dict) <= 0 else temp_dict

    # Insert Vaccine Stocks
    def insert_vaccine_stocks(self, cell_pkey, design_pkey, stock_count):
        """
        INSERT INTO VACCINE_STOCKS (CELL_PKEY, DESIGN_PKEY, STOCK_COUNT)
        VALUES ({cell_pkey}, {design_pkey}, {stock_count})
        """
        stock = models.VaccineStock(cell_pkey=cell_pkey, design_pkey=design_pkey, stock_count=stock_count)
        self.session.add(stock)

    # Delete Vaccine Stocks
    def delete_vaccine_stocks(self, cell_pkey):
        """
        DELETE FROM VACCINE_STOCKS WHERE CELL_PKEY = {cell_pkey}
        """
        self.session.query(models.VaccineStock).filter_by(cell_pkey=cell_pkey).delete()

    # 백신 companyId, storeId, deviceId로 모든 칸의 POG 상품 정보 가져오기
    def get_vaccine_pog_goods(self, company_id, store_id, device_id):
        """
        SELECT SHELVES.SHELF_FLOOR, VACCINE_CELLS.CELL_COLUMN, GOODS.GOODS_NAME, GOODS.GOODS_ID
        FROM COMPANIES
        LEFT JOIN STORES ON COMPANIES.COMPANY_PKEY = STORES.COMPANY_PKEY
        LEFT JOIN DEVICES ON STORES.STORE_PKEY = DEVICES.STORE_PKEY
        LEFT JOIN SHELVES ON SHELVES.DEVICE_PKEY = DEVICES.DEVICE_PKEY
        LEFT JOIN VACCINE_CELLS ON SHELVES.SHELF_PKEY = VACCINE_CELLS.SHELF_PKEY
        LEFT JOIN DESIGNS ON  DESIGNS.DESIGN_PKEY = VACCINE_CELLS.DESIGN_PKEY_MASTER
        LEFT JOIN GOODS ON GOODS.GOODS_ID = DESIGNS.GOODS_ID
        WHERE COMPANIES.COMPANY_ID = '{company_id}' AND STORES.STORE_ID = '{store_id}' AND DEVICES.DEVICE_ID = '{device_id}'
        ORDER BY SHELVES.SHELF_FLOOR, VACCINE_CELLS.CELL_COLUMN
        """
        result = \
            self.session.query(models.Shelf.shelf_floor, models.VaccineCell.cell_column, \
                                models.Good.goods_name, models.Good.goods_id)\
            .select_from(models.Company)\
            .join(models.Store, models.Store.company_pkey==models.Company.company_pkey)\
            .join(models.Device, models.Device.store_pkey==models.Store.store_pkey)\
            .join(models.Shelf, models.Shelf.device_pkey==models.Device.device_pkey)\
            .join(models.VaccineCell, models.VaccineCell.shelf_pkey==models.Shelf.shelf_pkey)\
            .join(models.Design, models.VaccineCell.design_pkey_master == models.Design.design_pkey)\
            .join(models.Good, models.Good.goods_id == models.Design.goods_id)\
            .filter(models.Company.company_id==company_id,
                    models.Store.store_id==store_id,
                    models.Device.device_id==device_id)\
            .order_by(models.Shelf.shelf_floor,
                    models.VaccineCell.cell_column)
        df = pd.read_sql(result.statement, self.session.bind)
        return df

    # Insert Vaccine Trade Log
    def insert_vaccine_trade_log(
        self,
        vaccine_trade_log_no,
        vaccine_trade_log_date,
        vaccine_trade_log_time,
        company_id,
        store_id,
        device_id,
        shelf_floor,
        cell_column,
        goods_id,
        goods_name,
        goods_label,
        goods_count,
        stock_left,
        duration,
        work_user,
        work_type,
        status_code,
        total_cnt,
        sale_price,
        total_sale_price
    ):
        i = insert(self.vaccine_trade_log)
        i = i.values({
            "vaccine_trade_log_no": vaccine_trade_log_no,
            "vaccine_trade_log_date": vaccine_trade_log_date,
            "vaccine_trade_log_time": vaccine_trade_log_time,
            "company_id": company_id,
            "store_id": store_id,
            "device_id": device_id,
            "shelf_floor": shelf_floor,
            "cell_column": cell_column,
            "goods_id": goods_id,
            "goods_name": goods_name,
            "goods_label": goods_label,
            "goods_count": goods_count,
            "stock_left": stock_left,
            "duration": duration,
            "work_user": work_user,
            "work_type": work_type,
            "status_code": status_code,
            "total_cnt": total_cnt,
            "sale_price": sale_price,
            "total_sale_price": total_sale_price})
        self.session.execute(i)

if __name__ == '__main__':
    pass
    # from sqlalchemy.orm import sessionmaker, aliased
    # from keys import keys
    # print(dao.insert_trade_log('0001', '00888', 's_00009', 0, 0, '8801037039993', '티오피)아메리카노200ml', 'top_americano_200_can',
    #                            1, 216.2, 21.62, 271, 52, 0, 'customer', 'trade', '000'))