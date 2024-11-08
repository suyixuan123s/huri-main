import socket
import struct

#H: 2 byte unsign integer B 1 byte unsign integer f 4 byte float
# data = struct.pack("HHf",8,40,.8)
num_parameter = 7
data = struct.pack("HH"+"f"*num_parameter,7, 7,*[0.0, 0.0, 123.0, 1.0, 0.0, 0.0, 0.0])
print(data)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(5)
client.connect(("192.168.125.1", 5001))

print('Socket successfully opened!')

client.send(data)

import time
time.sleep(100)