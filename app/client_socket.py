# client_socket.py
import time

from socket import *

class Client_Socket:
    def __init__(self, ip='127.0.0.1', port=8080):
        self.ip = ip
        self.port = port
        self.client_sock = None

    # Socket Conn
    def Connect(self):
        try:
            self.client_sock = socket(AF_INET, SOCK_STREAM)
            self.client_sock.settimeout(3) # socket Timeout Setting (3 sec)
            self.client_sock.connect((self.ip, self.port))
        except Exception as e:
            logmsg = f'[Connect] Error ({str(e)})'
            return False, logmsg
        else:
            logmsg = f'[Connect] ip({self.ip}) / port({self.port}) Connect Success.'
            return True, logmsg

    # Socket DisConnection
    def DisConnect(self):
        try:
            self.client_sock.close()
        except Exception as e:
            logmsg = f'[DisConnect] Error ({str(e)})'
        else: 
            logmsg = f'[DisConnect] ip({self.ip}) / port({self.port}) DisConnect Success.'
        return logmsg

    # Socket ReConnection
    def ReConnect(self):
        toBreak = False
        toCount = 0
        # 최대 3번 Retry conn
        while True:
            self.client_sock.close()
            try:
                self.client_sock.connect((self.ip, self.port))
                toBreak = True
            except Exception as e:
                print(f'[ReConnect] Error ({str(e)})')
            if toBreak:
                break
            if toCount == 3:
                logmsg = self.DisConnect()
                break

            toCount += 1
            time.sleep(1)

    # Send Msg
    def sendMsg(self, msg):
        try:
            self.client_sock.sendall(msg)
        except Exception as e:
            logmsg = f'[sendMsg] Error ({str(e)})'
            self.DisConnect()
            return False, logmsg
        else:
            logmsg = f'[sendMsg] > "{msg}" send Success.'
            return True, logmsg

    # Recv Msg
    def recvMsg(self):
        while True:
            try:
                recvdata = self.client_sock.recv(1024)
                if not recvdata:
                    logmsg = self.DisConnect()
                    return False, logmsg
            except Exception as e:
                logmsg = f'[recvMsg] Error ({str(e)})'
                return False, logmsg
            else:
                return True, recvdata