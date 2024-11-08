""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230706osaka

"""

if __name__ == '__main__':
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('100.80.147.16', 9999))
    s.listen(1)

    conn, addr = s.accept()
    print('Connected by', addr)

    while True:
        data = input('(Pdb) ')
        print(f"Send data is {data}")
        conn.sendall(data.encode())
        if not data:
            break

    conn.close()