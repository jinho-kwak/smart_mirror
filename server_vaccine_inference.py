import os
from app import create_app_vaccine
from app.config import Config
# from dotenv import load_dotenv

# load_dotenv()

app = create_app_vaccine(Config.BOILERPLATE_ENV)
app.debug = False
app.app_context().push()
if __name__ == "__main__":
    if Config.BOILERPLATE_ENV == 'dev':
        app.run(host='0.0.0.0', port=7100)
    else:
        app.run(port=7100)