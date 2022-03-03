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


# #-*- coding:utf-8 -*-
# import logging

# class Message:
#     def __init__(self, fmt, args):
#         self.fmt = fmt
#         self.args = args

#     def __str__(self):
#         return self.fmt.format(*self.args)

# class StyleAdapter(logging.LoggerAdapter):
#     """
#         logging.info("format this message {0}", 1)
#         logging.info("format this message {0}".format(1))
#         등을 지원하기 위한 클래스
#         기본 logging을 사용하고 위와 같은 포맷을 사용하면 에러가 난다.
#         StyleAdapter 클래스를 써서 위와같은 로깅 포맷을 사용할 수 있다.
#     """
#     def __init__(self, logger, extra=None):
#         super(StyleAdapter, self).__init__(logger, extra or {})

#     def log(self, level, msg, *args, **kwargs):
#         if self.isEnabledFor(level):
#             msg, kwargs = self.process(msg, kwargs)
#             self.logging._log(level, Message(msg, args), (), **kwargs)
    


# logger = StyleAdapter(logging.getLogger(__name__))

# def main():
#     logging.debug('Hello, {}', 'world!')

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     main()