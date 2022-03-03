import os
import ssl

from app import create_app_main
from app.config import Config

app = create_app_main(Config.BOILERPLATE_ENV)
app.app_context().push()


# if Config.UES_SSL == "True":                
#     context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
#     context.load_cert_chain(Config.SSL_CRT_PATH, Config.SSL_KEY_PATH) 
#     app.run(host='0.0.0.0', port=5000, ssl_context=(context))
# else:
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
