import serial
import time

class Sender:
    TERMINATOR = '\r'.encode('UTF8')

    def __init__(self, device='/dev/ttyACM0', baud=9600, timeout=1):
        self.serial = serial.Serial(device, baud, timeout=timeout)

    def receive(self) -> str:
        line = self.serial.read_until(self.TERMINATOR)
        return line.decode('UTF8').strip()

    def wait_callback(self, time_out =5):
        time_start = time.time()
        while True:
            data = self.receive()
            if data == "OK":
                time.sleep(.001)
                print(data)
                break
            if time.time()-time_start > time_out:
                print("TIMEOUT and NO DATA Receive")
                break
            print(data)

    def send(self, text: str) -> bool:
        line = '%s\r\f' % text
        self.serial.write(line.encode('UTF8'))
        # the line should be echoed.
        # If it isn't, something is wrong.
        return text == self.receive()

    def close(self):
        self.serial.close()


s = Sender(device='COM4')
print(s.send('test()'))
s.wait_callback(time_out=1)