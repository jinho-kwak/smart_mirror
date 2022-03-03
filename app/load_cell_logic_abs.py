#-*-coding utf-8-*-
import numpy as np

class Counter:
    def __init__(self, design_pkey_master, dao, p_rate, p_else_rate):
        self.design_pkey_master = design_pkey_master
        self.change = {}
        self.stock = {}
        self.dao = dao
        self.master_mean_weight, self.master_std_weight = self.dao.get_design_mean_n_std_weight(design_pkey_master)
        self.PERMIT_RATE = p_rate
        self.PERMIT_ELSE_RATE = p_else_rate
        self.ADMIN_LOG_WEIGHT = None
        self.ADMIN_LOG_COUNT = None
        self.empty_scale = 0.41         # empty 모드를 'lc'로 적용할 때, 최소 무게 배율
        self.empty_limit_weight = 170   # empty 모드를 'lc'로 적용할 수 있는 상품 최소 무게 limit (단위 : g)
        
    
    def count(self,
              weight_at_close,
              stock_at_open,
              design_pkey_inf_main,
              design_pkey_inf_empty,
              inference_mode,
              empty_mode,
              **kwargs):
        status = True
        if inference_mode == 'mix':
            # empty 모드를 'lc'로 적용 할 수 있는 최소 무게보다 작으면 강제 'cv' 변환
            if self.master_mean_weight < self.empty_limit_weight and empty_mode == 'lc':
                empty_mode = 'cv'

            self.empty = np.logical_or(
                empty_mode == 'lc' and weight_at_close <= int(self.master_mean_weight) * self.empty_scale,
                empty_mode == 'cv' and design_pkey_inf_empty == 0,
            )
            if self.empty:
                front_mean_weight = 0
                front_count = 0
            else:
                front_mean_weight, front_std_weight = self.dao.get_design_mean_n_std_weight(design_pkey_inf_main)
                front_count = 1
            
            # get count
            status, master_count = self.get_count(weight_at_close - front_mean_weight)
            
            # get stock
            self.stock[self.design_pkey_master] = master_count
            if front_count > 0:
                self.stock[design_pkey_inf_main] = self.stock.get(design_pkey_inf_main, 0) + front_count

            # get change
            self.change = stock_at_open.copy()
            for k, v in self.stock.items():
                self.change[k] = self.change.get(k, 0) - v
                if self.change[k] == 0:
                    del self.change[k]
            
            # delete zero master case
            if self.change.get(self.design_pkey_master) == 0:
                del self.change[self.design_pkey_master]

            admin_log_info = {
                'permit_rate' : self.PERMIT_RATE,
                'permit_else_rate' : self.PERMIT_ELSE_RATE,
                'admin_log_weight' : self.ADMIN_LOG_WEIGHT,
                'admin_log_count' : self.ADMIN_LOG_COUNT
            }
            return status, admin_log_info
            
        elif inference_mode=='lc':
            # get count
            status, master_count = self.get_count(weight_at_close)

            # get stock
            self.stock[self.design_pkey_master] = master_count

            # get change
            self.change = stock_at_open.copy()
            for k, v in self.stock.items():
                self.change[k] = self.change.get(k, 0) - v
                if self.change[k] == 0:
                    del self.change[k]
            
            # delete zero master case
            if self.change.get(self.design_pkey_master) == 0:
                del self.change[self.design_pkey_master]

            admin_log_info = {
                'permit_rate' : self.PERMIT_RATE,
                'permit_else_rate' : self.PERMIT_ELSE_RATE,
                'admin_log_weight' : self.ADMIN_LOG_WEIGHT,
                'admin_log_count' : self.ADMIN_LOG_COUNT
            }
            return status, admin_log_info
    
    def get_count(self, weight):
        count = round(weight / self.master_mean_weight)
        self.ADMIN_LOG_WEIGHT = weight
        self.ADMIN_LOG_COUNT = count
        control_limit = self.master_mean_weight * self.PERMIT_RATE.get(count, self.PERMIT_ELSE_RATE)
        abs_weight_error = abs(weight - count * self.master_mean_weight)
        if abs_weight_error > min(control_limit, self.master_mean_weight * 0.5):
            return False, count
        else:
            return True, count
