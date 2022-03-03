#-*-coding utf-8-*-
import numpy as np

class Counter:
    def __init__(self, design_pkey, dao, p_rate, p_else_rate):
        self.design_pkey = design_pkey
        self.change = {}
        self.stock = {}
        self.dao = dao
        self.master_mu, self.master_sigma = self.dao.get_design_mean_n_std_weight(design_pkey)
        self.PERMIT_RATE = p_rate
        self.PERMIT_ELSE_RATE = p_else_rate
        self.ADMIN_LOG_WEIGHT = None
        self.ADMIN_LOG_COUNT = None
    
    def count(self,
              weight_at_open,
              weight_at_close,
              stock_at_open,
              design_pkey_at_open,
              design_pkey_inf_main,
              design_pkey_inf_empty,
              inference_mode,
              empty_mode,
              **kwargs):
        status = True
        
        w_o = weight_at_open
        w_c = weight_at_close
        
        '''
        # error case
        if w_c - w_o < -20 and head_design_at_open is None:
            return False
        elif w_c - w_o > 20 and head_design_at_close is None:
            return False
        elif abs(w_c - w_o) < 20 and\
             head_design_at_open is not None and\
             head_design_at_close is None:
            return False
        elif abs(w_c - w_o) < 20 and\
             head_design_at_open is None and\
             head_design_at_close is not None:
            return False
        '''
        
        if inference_mode=='mix':
            if design_pkey_at_open is None or np.isnan(design_pkey_at_open):
                fw_o = 0
                fn_o = 0
            else:
                fw_o, _ = self.dao.get_design_mean_n_std_weight(design_pkey_at_open)
                fn_o = 1

            self.empty = design_pkey_inf_empty == 0
            if self.empty:
                self.empty = True
                fw_c = 0
                fn_c = 0
            else:
                fw_c, _ = self.dao.get_design_mean_n_std_weight(design_pkey_inf_main)
                fn_c = 1
            
            # test count
            bn = int(self.count_pick_of_back(w_o, w_c, fw_o, fw_c))
            cl = self.master_mu * self.PERMIT_RATE.get(bn, self.PERMIT_ELSE_RATE)
            target = w_o - fw_o - (w_c - fw_c) - self.master_mu * bn
            self.ADMIN_LOG_WEIGHT = w_o - fw_o - (w_c - fw_c)
            self.ADMIN_LOG_COUNT = bn
            
            if abs(target) > min(cl * max(1, bn), self.master_mu * 0.5):
                status = False
            
            # count change
            change = {}
            change[self.design_pkey] = bn
            if design_pkey_at_open is not None:
                change[design_pkey_at_open] = change.get(design_pkey_at_open, 0) + fn_o
            if design_pkey_inf_empty == 1:
                change[design_pkey_inf_main] = change.get(design_pkey_inf_main, 0) - fn_c
            for key, value in change.copy().items():
                if value == 0:
                    del change[key]
            self.change = change
            
            # count stock
            stock = stock_at_open
            for key, value in self.change.items():
                stock[key] = stock.get(key, 0) - value
                if stock[key] == 0:
                    del stock[key]
            if stock.get(self.design_pkey, 0) == 0:
                stock[self.design_pkey] = 0
            self.stock = stock

            admin_log_info = {
                'permit_rate' : self.PERMIT_RATE,
                'permit_else_rate' : self.PERMIT_ELSE_RATE,
                'admin_log_weight' : self.ADMIN_LOG_WEIGHT,
                'admin_log_count' : self.ADMIN_LOG_COUNT
            }
            
            return status, admin_log_info
            
        elif inference_mode=='lc':
            n = int(round((w_o - w_c) / self.master_mu))
            cl = self.master_mu * self.PERMIT_RATE.get(n, self.PERMIT_ELSE_RATE)
            target = w_o - w_c - self.master_mu * n
            self.ADMIN_LOG_WEIGHT = w_o - w_c
            self.ADMIN_LOG_COUNT = n

            if abs(target) > min(cl * max(1, n), self.master_mu * 0.5):
                status = False
            
            # count change
            if n != 0:
                self.change = {self.design_pkey: n}
            
            # count stock
            stock = stock_at_open
            for key, value in self.change.items():
                stock[key] = stock.get(key, 0) - value
                if stock[key] == 0:
                    del stock[key]
            if stock.get(self.design_pkey, 0) == 0:
                stock[self.design_pkey] = 0
            self.stock = stock

            admin_log_info = {
                'permit_rate' : self.PERMIT_RATE,
                'permit_else_rate' : self.PERMIT_ELSE_RATE,
                'admin_log_weight' : self.ADMIN_LOG_WEIGHT,
                'admin_log_count' : self.ADMIN_LOG_COUNT
            }
            
            return status, admin_log_info
    
    def count_pick_of_back(self, w_o, w_c, fw_o, fw_c):
        bw_o = w_o - fw_o
        bw_c = w_c - fw_c
        return round((bw_o - bw_c) / self.master_mu)
