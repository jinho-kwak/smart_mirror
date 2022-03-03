# coding: utf-8
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()



class Cell(db.Model):
    __tablename__ = 'cells'
    __table_args__ = (
        db.UniqueConstraint('shelf_pkey', 'cell_column'),
    )

    cell_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    shelf_pkey = db.Column(db.ForeignKey('shelves.shelf_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey_master = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    design_pkey_front = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    cell_column = db.Column(db.SmallInteger, nullable=False)
    stock_count_max = db.Column(db.SmallInteger)
    inference_mode = db.Column(db.String(3))
    load_cell_mode = db.Column(db.String(3))
    stock_count_low_alert_limit = db.Column(db.SmallInteger)
    empty_mode = db.Column(db.String(4), server_default=db.FetchedValue())

    design = db.relationship('Design', primaryjoin='Cell.design_pkey_front == Design.design_pkey', backref='design_cells')
    design1 = db.relationship('Design', primaryjoin='Cell.design_pkey_master == Design.design_pkey', backref='design_cells_0')
    shelf = db.relationship('Shelf', primaryjoin='Cell.shelf_pkey == Shelf.shelf_pkey', backref='cells')



class CigarCell(db.Model):
    __tablename__ = 'cigar_cells'
    __table_args__ = (
        db.UniqueConstraint('shelf_pkey', 'cell_column'),
    )

    cell_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    shelf_pkey = db.Column(db.ForeignKey('shelves.shelf_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey_master = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    design_pkey_first = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    design_pkey_second = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    design_pkey_third = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    cell_column = db.Column(db.SmallInteger, nullable=False)
    stock_count_max = db.Column(db.SmallInteger)
    stock_count_low_alert_limit = db.Column(db.SmallInteger)
    total_cnt = db.Column(db.Integer)
    last_upd = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    inference_mode = db.Column(db.String(3))
    design = db.relationship('Design', primaryjoin='CigarCell.design_pkey_first == Design.design_pkey', backref='design_cigar_cells')
    design1 = db.relationship('Design', primaryjoin='CigarCell.design_pkey_master == Design.design_pkey', backref='design_cigar_cells_0')
    design2 = db.relationship('Design', primaryjoin='CigarCell.design_pkey_second == Design.design_pkey', backref='design_cigar_cells_1')
    design3 = db.relationship('Design', primaryjoin='CigarCell.design_pkey_third == Design.design_pkey', backref='design_cigar_cells_2')
    shelf = db.relationship('Shelf', primaryjoin='CigarCell.shelf_pkey == Shelf.shelf_pkey', backref='cigar_cells')



class CigarStock(db.Model):
    __tablename__ = 'cigar_stocks'
    __table_args__ = (
        db.UniqueConstraint('cell_pkey', 'design_pkey'),
    )

    stock_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    cell_pkey = db.Column(db.ForeignKey('cigar_cells.cell_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), nullable=False, index=True)
    stock_count = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())

    cigar_cell = db.relationship('CigarCell', primaryjoin='CigarStock.cell_pkey == CigarCell.cell_pkey', backref='cigar_stocks')
    design = db.relationship('Design', primaryjoin='CigarStock.design_pkey == Design.design_pkey', backref='cigar_stocks')



class CigarTradeLog(db.Model):
    __tablename__ = 'cigar_trade_log'

    cigar_trade_log_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    cigar_trade_log_no = db.Column(db.BigInteger)
    cigar_trade_log_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    cigar_trade_log_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10))
    store_id = db.Column(db.String(10))
    device_id = db.Column(db.String(10))
    shelf_floor = db.Column(db.SmallInteger)
    cell_column = db.Column(db.SmallInteger)
    goods_id = db.Column(db.String(13))
    goods_name = db.Column(db.String(50))
    goods_label = db.Column(db.String(50))
    goods_count = db.Column(db.SmallInteger)
    stock_left = db.Column(db.String(1000))
    duration = db.Column(db.Float(53))
    work_user = db.Column(db.String(20))
    work_type = db.Column(db.String(10))
    status_code = db.Column(db.String(5))
    total_cnt = db.Column(db.SmallInteger)
    sale_price = db.Column(db.Integer)
    total_sale_price = db.Column(db.Integer)



class Company(db.Model):
    __tablename__ = 'companies'

    company_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10), nullable=False, unique=True)
    company_name = db.Column(db.String(50))
    operation = db.Column(db.Boolean, nullable=False)
    partner_pkey = db.Column(db.ForeignKey('partners.partner_pkey'))

    partner = db.relationship('Partner', primaryjoin='Company.partner_pkey == Partner.partner_pkey', backref='companies')



class DesignTagLink(db.Model):
    __tablename__ = 'design_tag_link'
    __table_args__ = (
        db.UniqueConstraint('design_tag_pkey', 'tag_pkey', 'model_pkey'),
    )

    design_tag_link_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    design_tag_pkey = db.Column(db.Integer, nullable=False)
    tag_pkey = db.Column(db.ForeignKey('tag.tag_pkey', ondelete='CASCADE'), nullable=False)
    model_pkey = db.Column(db.ForeignKey('models.model_pkey', ondelete='CASCADE'), nullable=False)

    model = db.relationship('Model', primaryjoin='DesignTagLink.model_pkey == Model.model_pkey', backref='design_tag_links')
    tag = db.relationship('Tag', primaryjoin='DesignTagLink.tag_pkey == Tag.tag_pkey', backref='design_tag_links')



class Design(db.Model):
    __tablename__ = 'designs'

    design_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    goods_id = db.Column(db.ForeignKey('goods.goods_id'))
    design_type = db.Column(db.String(20))
    design_mean_weight = db.Column(db.Float(53))
    design_std_weight = db.Column(db.Float(53))
    design_infer_label = db.Column(db.String(50), nullable=False, unique=True)
    design_img_url = db.Column(db.String(2048))
    design_tag_pkey = db.Column(db.Integer)

    goods = db.relationship('Good', primaryjoin='Design.goods_id == Good.goods_id', backref='designs')



class Device(db.Model):
    __tablename__ = 'devices'

    device_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    device_id = db.Column(db.String(10), nullable=False)
    device_install_type = db.Column(db.String(1))
    device_storage_type = db.Column(db.String(2))
    operation = db.Column(db.Boolean, nullable=False)
    store_pkey = db.Column(db.ForeignKey('stores.store_pkey'))
    alarm = db.Column(db.Boolean)
    store_convert_operation = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())
    admin_pog_change = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())

    store = db.relationship('Store', primaryjoin='Device.store_pkey == Store.store_pkey', backref='devices')



class Event(db.Model):
    __tablename__ = 'events'

    event_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    event_name = db.Column(db.String(50))
    event_type = db.Column(db.String(50))
    event_description = db.Column(db.String(1000))



t_events_discount = db.Table(
    'events_discount',
    db.Column('event_pkey', db.BigInteger, nullable=False, server_default=db.FetchedValue()),
    db.Column('event_name', db.String(50)),
    db.Column('event_type', db.String(50)),
    db.Column('event_description', db.String(1000)),
    db.Column('discount_rate', db.Float(53))
)



t_events_npone = db.Table(
    'events_npone',
    db.Column('event_pkey', db.BigInteger, nullable=False, server_default=db.FetchedValue()),
    db.Column('event_name', db.String(50)),
    db.Column('event_type', db.String(50)),
    db.Column('event_description', db.String(1000)),
    db.Column('condition_count', db.Integer)
)



class Good(db.Model):
    __tablename__ = 'goods'

    goods_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    goods_id = db.Column(db.String(13), nullable=False, unique=True)
    goods_name = db.Column(db.String(50))



class LcModeGoodsList(db.Model):
    __tablename__ = 'lc_mode_goods_list'
    __table_args__ = (
        db.UniqueConstraint('goods_id', 'model_name'),
    )

    lc_mode_goods_list_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    model_name = db.Column(db.String(20))
    goods_id = db.Column(db.ForeignKey('goods.goods_id'), nullable=False)

    goods = db.relationship('Good', primaryjoin='LcModeGoodsList.goods_id == Good.goods_id', backref='lc_mode_goods_lists')



class Loadcell(db.Model):
    __tablename__ = 'loadcells'
    __table_args__ = (
        db.UniqueConstraint('shelf_pkey', 'cell_pkey', 'loadcell_column'),
    )

    loadcell_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    shelf_pkey = db.Column(db.ForeignKey('shelves.shelf_pkey', ondelete='CASCADE'))
    cell_pkey = db.Column(db.ForeignKey('cells.cell_pkey', ondelete='SET NULL'))
    loadcell_column = db.Column(db.SmallInteger, nullable=False)

    cell = db.relationship('Cell', primaryjoin='Loadcell.cell_pkey == Cell.cell_pkey', backref='loadcells')
    shelf = db.relationship('Shelf', primaryjoin='Loadcell.shelf_pkey == Shelf.shelf_pkey', backref='loadcells')



class Model(db.Model):
    __tablename__ = 'models'

    model_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    model_name = db.Column(db.String(20))
    model_address = db.Column(db.String(2083))
    model_type = db.Column(db.String(4))



class Partner(db.Model):
    __tablename__ = 'partners'

    partner_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    partner_id = db.Column(db.String(10), nullable=False, unique=True)
    partner_name = db.Column(db.String(50))
    operation = db.Column(db.Boolean, nullable=False)



class Sale(db.Model):
    __tablename__ = 'sales'

    sale_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    sale_reg_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    sale_reg_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    sale_price = db.Column(db.Integer, nullable=False)
    store_pkey = db.Column(db.ForeignKey('stores.store_pkey', ondelete='CASCADE'))
    design_pkey = db.Column(db.ForeignKey('designs.design_pkey', ondelete='CASCADE'))
    event_pkey = db.Column(db.ForeignKey('events.event_pkey'))

    design = db.relationship('Design', primaryjoin='Sale.design_pkey == Design.design_pkey', backref='sales')
    event = db.relationship('Event', primaryjoin='Sale.event_pkey == Event.event_pkey', backref='sales')
    store = db.relationship('Store', primaryjoin='Sale.store_pkey == Store.store_pkey', backref='sales')



class Shelf(db.Model):
    __tablename__ = 'shelves'
    __table_args__ = (
        db.UniqueConstraint('device_pkey', 'shelf_floor'),
    )

    shelf_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    device_pkey = db.Column(db.ForeignKey('devices.device_pkey', ondelete='CASCADE'), nullable=False)
    shelf_floor = db.Column(db.SmallInteger, nullable=False)
    shelf_storage_type = db.Column(db.String(4))
    model_pkey = db.Column(db.ForeignKey('models.model_pkey', ondelete='SET NULL', onupdate='CASCADE'))
    object_pkey = db.Column(db.Integer)

    device = db.relationship('Device', primaryjoin='Shelf.device_pkey == Device.device_pkey', backref='shelves')
    model = db.relationship('Model', primaryjoin='Shelf.model_pkey == Model.model_pkey', backref='shelves')



class Stock(db.Model):
    __tablename__ = 'stocks'
    __table_args__ = (
        db.UniqueConstraint('cell_pkey', 'design_pkey'),
    )

    stock_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    cell_pkey = db.Column(db.ForeignKey('cells.cell_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), nullable=False, index=True)
    stock_count = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())

    cell = db.relationship('Cell', primaryjoin='Stock.cell_pkey == Cell.cell_pkey', backref='stocks')
    design = db.relationship('Design', primaryjoin='Stock.design_pkey == Design.design_pkey', backref='stocks')



class Store(db.Model):
    __tablename__ = 'stores'

    store_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    store_id = db.Column(db.String(10), nullable=False, unique=True)
    store_name = db.Column(db.String(50))
    operation = db.Column(db.Boolean, nullable=False)
    company_pkey = db.Column(db.ForeignKey('companies.company_pkey'))

    company = db.relationship('Company', primaryjoin='Store.company_pkey == Company.company_pkey', backref='stores')



class Tag(db.Model):
    __tablename__ = 'tag'

    tag_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    tag_value = db.Column(db.String(50), nullable=False, unique=True)



class TradeLog(db.Model):
    __tablename__ = 'trade_log'

    trade_log_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    trade_log_no = db.Column(db.BigInteger)
    trade_log_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    trade_log_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10))
    store_id = db.Column(db.String(10))
    device_id = db.Column(db.String(10))
    shelf_floor = db.Column(db.SmallInteger)
    cell_column = db.Column(db.SmallInteger)
    goods_id = db.Column(db.String(13))
    goods_name = db.Column(db.String(50))
    goods_label = db.Column(db.String(50))
    goods_count = db.Column(db.SmallInteger)
    stock_left = db.Column(db.String(1000))
    goods_mean_weight = db.Column(db.Float(53))
    goods_std_weight = db.Column(db.Float(53))
    open_weight = db.Column(db.SmallInteger)
    close_weight = db.Column(db.SmallInteger)
    duration = db.Column(db.Float(53))
    work_user = db.Column(db.String(20))
    work_type = db.Column(db.String(10))
    status_code = db.Column(db.String(5))
    total_cnt = db.Column(db.SmallInteger)
    sale_price = db.Column(db.Integer)
    total_sale_price = db.Column(db.Integer)


t_trade_log_partitions = db.Table(
    'trade_log_partitions',
    db.Column('parent_schema', db.String),
    db.Column('parent', db.String),
    db.Column('child_schema', db.String),
    db.Column('child', db.String)
)



class Trade(db.Model):
    __tablename__ = 'trades'

    trade_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    trade_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    trade_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    device_pkey = db.Column(db.ForeignKey('devices.device_pkey', ondelete='CASCADE'), nullable=False)

    device = db.relationship('Device', primaryjoin='Trade.device_pkey == Device.device_pkey', backref='trades')



class VaccineCell(db.Model):
    __tablename__ = 'vaccine_cells'
    __table_args__ = (
        db.UniqueConstraint('shelf_pkey', 'cell_column'),
    )

    cell_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    shelf_pkey = db.Column(db.ForeignKey('shelves.shelf_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey_master = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), index=True)
    cell_column = db.Column(db.SmallInteger, nullable=False)
    stock_count_max = db.Column(db.SmallInteger)
    stock_count_low_alert_limit = db.Column(db.SmallInteger)
    total_cnt = db.Column(db.Integer)
    last_upd = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    design = db.relationship('Design', primaryjoin='VaccineCell.design_pkey_master == Design.design_pkey', backref='vaccine_cells')
    shelf = db.relationship('Shelf', primaryjoin='VaccineCell.shelf_pkey == Shelf.shelf_pkey', backref='vaccine_cells')



class VaccineStock(db.Model):
    __tablename__ = 'vaccine_stocks'
    __table_args__ = (
        db.UniqueConstraint('cell_pkey', 'design_pkey'),
    )

    stock_pkey = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    cell_pkey = db.Column(db.ForeignKey('vaccine_cells.cell_pkey', ondelete='CASCADE'), nullable=False)
    design_pkey = db.Column(db.ForeignKey('designs.design_pkey', ondelete='SET NULL'), nullable=False, index=True)
    stock_count = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())

    vaccine_cell = db.relationship('VaccineCell', primaryjoin='VaccineStock.cell_pkey == VaccineCell.cell_pkey', backref='vaccine_stocks')
    design = db.relationship('Design', primaryjoin='VaccineStock.design_pkey == Design.design_pkey', backref='vaccine_stocks')



class VaccineTradeCheck(db.Model):
    __tablename__ = 'vaccine_trade_check'

    vaccine_trade_check_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    vaccine_trade_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    vaccine_trade_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10))
    store_id = db.Column(db.String(10))
    device_id = db.Column(db.String(10))
    total_sale_price = db.Column(db.Integer)
    confirm_flag = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())
    confirm_user = db.Column(db.String(100))
    confirm_date = db.Column(db.Date)
    confirm_time = db.Column(db.Time(True))
    payment_flag = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())
    payment_date = db.Column(db.Date)
    payment_time = db.Column(db.Time(True))
    qr_data = db.Column(db.String(1000))
    user_level = db.Column(db.String(10))



class VaccineTradeLog(db.Model):
    __tablename__ = 'vaccine_trade_log'

    vaccine_trade_log_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    vaccine_trade_log_no = db.Column(db.BigInteger)
    vaccine_trade_log_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    vaccine_trade_log_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10))
    store_id = db.Column(db.String(10))
    device_id = db.Column(db.String(10))
    shelf_floor = db.Column(db.SmallInteger)
    cell_column = db.Column(db.SmallInteger)
    goods_id = db.Column(db.String(13))
    goods_name = db.Column(db.String(50))
    goods_label = db.Column(db.String(50))
    goods_count = db.Column(db.SmallInteger)
    stock_left = db.Column(db.String(1000))
    duration = db.Column(db.Float(53))
    work_user = db.Column(db.String(20))
    work_type = db.Column(db.String(10))
    status_code = db.Column(db.String(5))
    total_cnt = db.Column(db.SmallInteger)
    sale_price = db.Column(db.Integer)
    total_sale_price = db.Column(db.Integer)



class VaccineTradePog(db.Model):
    __tablename__ = 'vaccine_trade_pog'
    __table_args__ = (
        db.UniqueConstraint('vaccine_trade_check_pkey', 'vaccine_trade_no'),
    )

    vaccine_trade_pog_pkey = db.Column(db.BigInteger, primary_key=True, server_default=db.FetchedValue())
    vaccine_trade_check_pkey = db.Column(db.ForeignKey('vaccine_trade_check.vaccine_trade_check_pkey', ondelete='CASCADE'), nullable=False)
    vaccine_trade_no = db.Column(db.SmallInteger, nullable=False, server_default=db.FetchedValue())
    shelf_floor = db.Column(db.SmallInteger, nullable=False)
    cell_column = db.Column(db.SmallInteger, nullable=False)
    goods_id = db.Column(db.String(13), nullable=False)
    goods_name = db.Column(db.String(50))
    goods_label = db.Column(db.String(50))
    goods_count = db.Column(db.SmallInteger, nullable=False, server_default=db.FetchedValue())
    user_count = db.Column(db.SmallInteger)
    sale_price = db.Column(db.Integer, server_default=db.FetchedValue())
    vaccine_trade_date = db.Column(db.Date, nullable=False, server_default=db.FetchedValue())
    vaccine_trade_time = db.Column(db.Time(True), nullable=False, server_default=db.FetchedValue())
    company_id = db.Column(db.String(10))
    store_id = db.Column(db.String(10))
    device_id = db.Column(db.String(10))

    vaccine_trade_check = db.relationship('VaccineTradeCheck', primaryjoin='VaccineTradePog.vaccine_trade_check_pkey == VaccineTradeCheck.vaccine_trade_check_pkey', backref='vaccine_trade_pogs')
