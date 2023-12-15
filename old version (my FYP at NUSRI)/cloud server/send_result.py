# send back the signal

import numpy as np
predicted = np.load("/root/predicted.npy")
predicted = int(predicted)

import socket
import sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host='0.0.0.0'
port=6000
s.bind((host,port))
s.listen(5)
msg = chr(predicted+97)
while True:
    clientsocket, addr = s.accept()
    print("连接地址: %s" % str(addr))
    clientsocket.send(msg.encode('utf-8'))
    clientsocket.close()
