import huri.test.app.remote_exe_pb2_grpc as re_rpc
import huri.test.app.remote_exe_pb2 as re_msg
import grpc
import pickle
from huri.definitions.utils_structure import MotionBatch


class Remote_Exe_Client:
    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = re_rpc.RemoteExecuterStub(channel)

    def send_motion_batch(self, motion_batch: MotionBatch):
        self.stub.run_motion(re_msg.MotionBatch(
            data=pickle.dumps(motion_batch)
        ))

if __name__ == "__main__":
    mb_test = MotionBatch()
    client = Remote_Exe_Client()
    client.send_motion_batch(mb_test)