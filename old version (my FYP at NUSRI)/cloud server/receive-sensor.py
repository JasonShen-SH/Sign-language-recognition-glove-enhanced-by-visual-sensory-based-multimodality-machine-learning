# prepare for receiving sensor data

import socket
import numpy as np

tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

tcp_server.bind(("0.0.0.0", 8267))

tcp_server.listen(128)

tcp_client, tcp_client_address= tcp_server.accept()

print("client's IP and port: ",tcp_client_address)
print("client: ",tcp_client)

data = tcp_client.recv(1024)

data = data.decode(encoding='utf-8')
print("data received from client: ", data)

data = data.split(" ")

sensor_data=[]

for i in range(len(data)):
    if i==0:
        tmp=data[i][1:-1]
        tmp=float(tmp)
        sensor_data.append(tmp)
    else:
        tmp=data[i][:-1]
        tmp=float(tmp)
        sensor_data.append(tmp)
sensor_data = np.array(sensor_data)

np.save("/root/sensor_data.npy",sensor_data)

send_data = "thanks client! Your message is received".encode(encoding = "utf-8")
tcp_client.send(send_data)

tcp_client.close()
