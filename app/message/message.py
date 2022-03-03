import requests
from . import config
from . import auth

def sendMany(data):
  return requests.post(config.getUrl('/messages/v4/send-many'), headers=auth.get_headers(config.apiKey, config.apiSecret), json=data)

def sendOne(data):
  return requests.post(config.getUrl('/messages/v4/send'), headers=auth.get_headers(config.apiKey, config.apiSecret), json=data)
