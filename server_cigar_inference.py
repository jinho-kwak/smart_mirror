import os
from app import create_app_cigar
from app.config import Config
# from dotenv import load_dotenv

# load_dotenv()

app = create_app_cigar(Config.BOILERPLATE_ENV)
app.debug = False
app.app_context().push()
if __name__ == "__main__":
    if Config.BOILERPLATE_ENV == 'dev':
        app.run(host='0.0.0.0', port=7000)
    else:
        app.run(port=7000)