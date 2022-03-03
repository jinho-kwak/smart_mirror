#-*- coding: utf-8 -*-
import os
from keys import keys

# uncomment the line below for postgres database url from environment variable
# postgres_local_base = os.environ['DATABASE_URL']

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # BOILERPLATE_ENV choose : dev, prod, test
    BOILERPLATE_ENV = 'prod'
    AWS_S3 = 's3'
    AWS_S3_REGINON_NAME = 'ap-northeast-2'
    EMAIL_URL = 'https://3cdj0lxrud.execute-api.ap-northeast-2.amazonaws.com/Prod/send'
    SLACK_URL = {
                    'fridge':['https://hooks.slack.com/services/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',
                              'https://hooks.slack.com/services/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'],
                    'cigar':['https://hooks.slack.com/services/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'],
                    'vaccine':['https://hooks.slack.com/services/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'],
                }

    CAMERAS_LOCATION = {'NC':['l0','l1','r0','r1'],
                        'EC':['l0','l1','m0','m1','r0','r1'],
                        'VACCINE':['fl','fr','bl','br'],
                        }

    BLACK_IP = ["193.27.228.27", "45.129.33.120", "139.162.145.250", "192.241.235.203", "172.105.89.161", "195.37.190.77", "162.243.128.167"]
    WHITE_IP = ["125.132.250.212", "112.171.39.219", "172.18.0.1"]

class DevelopmentConfig(Config):
    # uncomment the line below to use postgres
    # SQLALCHEMY_DATABASE_URI = postgres_local_base

    # Redis
    REDIS_HOST = '125.132.250.212'
    REDIS_PORT = '6379'
    REDIS_DB = 2
    REDIS_USERNAME = 'default'
    REDIS_PASSWORD = keys.get('redis', './keys')
    REDIS_CHARSET = "utf-8"
    REDIS_DECODE_RESPONSES = True
    # Server
    INFERENCE_SERVER_HOST = '127.0.0.1'
    INFERENCE_FRIDGE_SERVER_PORT = '6000'
    INFERENCE_CIGAR_SERVER_PORT = '7000'
    INFERENCE_VACCINE_SERVER_PORT = '7100'

    #EC2_INFERENCE_IP = '10.0.1.146'
    EC2_INFERENCE_IP = '13.209.188.42'
    EC2_INFERENCE_PORT = '9000'
    EC2_OBJECT_DETECTION_IP = '183.111.67.201'
    EC2_OBJECT_DETECTION_PORT = '8000'
    # DB
    SQLALCHEMY_DATABASE_URI = f'postgresql://postgres:{keys.get("postgres", "./keys")}@smart-retail-db.ctnphj2dxhnf.ap-northeast-2.rds.amazonaws.com:5432/emart24'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # etc
    DEBUG = True
    EMAIL_ALARM = False
    GPU_MEMORY_LIMIT = 8000
    SAVE_LOG_TO_S3 = True


class TestingConfig(Config):
    # uncomment the line below to use postgres
    # SQLALCHEMY_DATABASE_URI = postgres_local_base

    # Redis
    REDIS_HOST = '127.0.0.1'
    REDIS_PORT = '6379'
    REDIS_DB = 2
    REDIS_USERNAME = 'default'
    REDIS_PASSWORD = keys.get('redis', './keys')
    REDIS_CHARSET = "utf-8"
    REDIS_DECODE_RESPONSES = True
    # Server
    INFERENCE_SERVER_HOST = '127.0.0.1'
    INFERENCE_FRIDGE_SERVER_PORT = '6000'
    INFERENCE_CIGAR_SERVER_PORT = '7000'
    EC2_INFERENCE_IP = '10.0.1.146'
    EC2_INFERENCE_PORT = '9000'
    # DB
    SQLALCHEMY_DATABASE_URI = f'postgresql://postgres:{keys.get("postgres", "./keys")}@smart-retail-db.ctnphj2dxhnf.ap-northeast-2.rds.amazonaws.com:5432/emart24'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # etc
    DEBUG = True
    EMAIL_ALARM = False
    GPU_MEMORY_LIMIT = 10000
    SAVE_LOG_TO_S3 = True

class ProductionConfig(Config):
    DEBUG = False
    EMAIL_ALARM = True
    # uncomment the line below to use postgres
    # SQLALCHEMY_DATABASE_URI = postgres_local_base

    # Redis
    REDIS_HOST = '125.132.250.212'
    REDIS_PORT = '6379'
    REDIS_DB = 2
    REDIS_USERNAME = 'default'
    REDIS_PASSWORD = keys.get('redis', './keys')
    REDIS_CHARSET = "utf-8"
    REDIS_DECODE_RESPONSES = True
    # Server
    INFERENCE_SERVER_HOST = '127.0.0.1'
    INFERENCE_FRIDGE_SERVER_PORT = '6000'
    INFERENCE_CIGAR_SERVER_PORT = '7000'
    INFERENCE_VACCINE_SERVER_PORT = '7100'

    EC2_INFERENCE_IP = '13.209.188.42'
    EC2_INFERENCE_PORT = '9000'
    EC2_OBJECT_DETECTION_IP = '183.111.67.201'
    EC2_OBJECT_DETECTION_PORT = '8000'

    # DB
    SQLALCHEMY_DATABASE_URI = f'postgresql://postgres:{keys.get("postgres", "./keys")}@smart-retail-db.ctnphj2dxhnf.ap-northeast-2.rds.amazonaws.com:5432/emart24'
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # etc
    DEBUG = False
    EMAIL_ALARM = True
    GPU_MEMORY_LIMIT = 10000
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    SAVE_LOG_TO_S3 = True


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

# key = Config.SECRET_KEY
