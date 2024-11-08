import grpc
import time
import pickle
from concurrent import futures
import numpy as np
import basis.trimesh as trm  # for creating obj
import modeling.geometric_model as gm
import modeling.model_collection as mc
import visualization.panda.world as wd
import robot_sim.robots.robot_interface as ri
import visualization.panda.rpc.rviz_server as rs
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
import huri.core.base_boost as bb
import cv2
from huri.core.common_import import *


def gen_load_detected_img(texture_show_yolo):
    def func():
        bb.set_img_texture(cv2.imread("yolo_tmp.jpg"), texture_show_yolo)

    return func


def serve(host="localhost:18300"):
    base = wd.World(cam_pos=[0, 2, 1.5], lookat_pos=[0, 0, .2])
    base = boost_base(base)
    base.render2, base.cam2, _ = base.spawn_window(base, win_size=(2064, 1544), name="render3")
    texture_show_yolo = bb.gen_img_texture_on_render(render=base.render2, img_shape=(2064, 1544))
    base.set_img_texture = gen_load_detected_img(texture_show_yolo)

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    rvs = rs.RVizServer()
    rv_rpc.add_RVizServicer_to_server(rvs, server)
    server.add_insecure_port(host)
    server.start()
    print("The RViz server is started!")
    base.run()


if __name__ == "__main__":
    serve(host="localhost:9999")
