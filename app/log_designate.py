import logging

class LogDesignate:
    def __init__(self, devkind = 'basic_log'):
        self.logger = None

        # 냉장고
        if devkind == 'fridge':
            self.logger = logging.getLogger('fridge_log')
        # 담배
        elif devkind == 'cigar':
            self.logger = logging.getLogger('cigar_log')
        # 주류
        elif devkind == 'alcohol':
            self.logger = logging.getLogger('alcohol_log')
        # 백신
        elif devkind == 'vaccine':
            self.logger = logging.getLogger('vaccine_log')
        # 상태체크
        elif devkind == 'check_status':
            self.logger = logging.getLogger('check_status_log')
        # 예외 (basic_log) 반영
        else:
            self.logger = logging.getLogger('basic_log')

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def exception(self, msg):
        self.logger.exception(msg)