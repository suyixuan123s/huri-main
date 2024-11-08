import time
import csv
import socket
import struct
import math


def bytes_to_fp32(bytes_data, is_big_endian=False):
    """
    bytes to float
    :param bytes_data: bytes
    :param is_big_endian: is big endian or not，default is False.
    :return: fp32
    """
    return struct.unpack('>f' if is_big_endian else '<f', bytes_data)[0]


def bytes_to_fp32_list(bytes_data, n=0, is_big_endian=False):
    """
    bytes to float list
    :param bytes_data: bytes
    :param n: quantity of parameters need to be converted，default is 0，all bytes converted.
    :param is_big_endian: is big endian or not，default is False.
    :return: float list
    """
    ret = []
    count = n if n > 0 else len(bytes_data) // 4
    for i in range(count):
        ret.append(bytes_to_fp32(bytes_data[i * 4: i * 4 + 4], is_big_endian))
    return ret


def bytes_to_u32(data):
    data_u32 = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3]
    return data_u32


robot_ip = '192.168.1.232'  # IP of controller
robot_port = 30002  # Port of controller

# create socket to connect controller
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setblocking(True)
sock.settimeout(1)
sock.connect((robot_ip, robot_port))

while True:

    data = sock.recv(4)
    length = bytes_to_u32(data)
    data += sock.recv(length - 4)
    speed_alljoints_rad = bytes_to_fp32_list(data[256:284])
    speed_alljoints_deg = []
    for i in range(len(speed_alljoints_rad)):
        speed_alljoints_deg.append(math.degrees(speed_alljoints_rad[i]))
    print(speed_alljoints_deg)
    with open('vel_joint.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([speed_alljoints_deg])
