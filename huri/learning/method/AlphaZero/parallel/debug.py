""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230706osaka

"""

if __name__ == '__main__':
    import select
    import socket
    import sys
    import threading

    host, port = '100.80.147.16', 54982
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    # s.settimeout(.5)
    read_sockets, write_sockets, error_sockets = select.select(
        [s], [], [],
    )


    def read_input(s: socket.socket):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            # print(line.strip())
            s.send(line.encode())


    t = threading.Thread(target=read_input, args=(s,))
    t.start()

    while True:
        for sock in read_sockets:
            if sock == s:
                # Incoming message from remote debugger.
                data = sock.recv(4096)
                if not data:
                    exit(0)
                else:
                    sys.stdout.write(data.decode())
                    sys.stdout.flush()
