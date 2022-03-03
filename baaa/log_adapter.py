#-*- coding: utf-8 -*-
import logging
import logging.config
import sys


# logger = logging.getLogger('console_timefile')


class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        
        # log_adapter.LoggerWriter object at 0x6c36efb0 에러가 떠서 주석
        # self.level(sys.stderr)
        pass

        
str_log_file_name = 'status.log'
logging.config.fileConfig("logging.conf", disable_existing_loggers=True, \
    defaults={"str_log_file_name" : str_log_file_name}) 
sys.stdout = LoggerWriter(logging.debug)
sys.stderr = LoggerWriter(logging.error)