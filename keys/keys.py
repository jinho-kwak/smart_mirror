from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
import os

def get(dbtype, key_path='./'):
    if dbtype == 'redis':
        filename = 'redis_key.bin'
    elif dbtype in ['postgres', 'postgresql']:
        filename = 'postgres_key.bin'
    
    file_in = open(os.path.join(key_path, filename), "rb")

    private_key = RSA.import_key(open(os.path.join(key_path, 'private.pem')).read())

    enc_session_key, nonce, tag, ciphertext = \
       [ file_in.read(x) for x in (private_key.size_in_bytes(), 16, 16, -1) ]

    # Decrypt the session key with the private RSA key
    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    # Decrypt the data with the AES session key
    cipher_aes = AES.new(session_key, AES.MODE_GCM, nonce)
    data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    return data.decode('utf-8')
