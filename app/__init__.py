from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import os

from . import models
from .config import config_by_name
import app.util as util
flask_bcrypt = Bcrypt()

# 메인 서버
def create_app_main(config_name):
    util.createFolder("./logs/log")
    app = Flask(__name__)

    from .route_main import bp
    app.register_blueprint(bp)

    app.config.from_object(config_by_name[config_name])
    models.db.init_app(app)
    flask_bcrypt.init_app(app)
    CORS(app) 
    
    return app

# 냉장고 인퍼런스 서버
def create_app_fridge(config_name):
    util.createFolder("./logs/log")
    app = Flask(__name__)

    from .route_fridge import fridge_bp
    app.register_blueprint(fridge_bp)

    app.config.from_object(config_by_name[config_name])
    models.db.init_app(app)
    flask_bcrypt.init_app(app)
    CORS(app) 

    return app

# 담배 인퍼런스 서버
def create_app_cigar(config_name):
    util.createFolder("./logs/log")
    app = Flask(__name__)

    from .route_cigar import cigar_bp
    app.register_blueprint(cigar_bp)

    app.config.from_object(config_by_name[config_name])
    models.db.init_app(app)
    flask_bcrypt.init_app(app)
    CORS(app)

    return app

# 백신 인퍼런스 서버
def create_app_vaccine(config_name):
    util.createFolder("./logs/log")
    app = Flask(__name__)

    from .route_vaccine import vaccine_bp
    app.register_blueprint(vaccine_bp)

    app.config.from_object(config_by_name[config_name])
    models.db.init_app(app)
    flask_bcrypt.init_app(app)
    CORS(app)

    return app

# def createFolder(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)