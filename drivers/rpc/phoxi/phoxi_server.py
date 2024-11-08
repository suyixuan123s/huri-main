import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

import drivers.devices.phoxi.phoxicontrol as pctrl
import drivers.rpc.phoxi.phoxi_pb2 as pxmsg
import drivers.rpc.phoxi.phoxi_pb2_grpc as pxrpc


class PhoxiServer(pxrpc.PhoxiServicer):

    def _unpackarraydata(self, dobj):
        h = dobj.width
        w = dobj.height
        ch = dobj.channel
        return np.frombuffer(dobj.image, (h, w, ch))

    def initialize(self, pctrlinstance, calibpath=""):

        """

        :param pctrlinstance: an instancde of driver.phoxi.phoxicontrol.pyd(PhoxiControl class)
        :return:
        """

        self._hasframe = False
        self._pcins = pctrlinstance
        # for read resolution of external camera
        self._calibpath = Path(calibpath)
        self._extcamimg_width = None
        self._extcamimg_height = None
        self._parse_calibpath()
        self._pcins.changeledpower(2000)

    def _parse_calibpath(self):
        if self._calibpath.exists() and self._calibpath.name.endswith("txt"):
            with self._calibpath.open() as f:
                data = f.read().lstrip().rstrip().split("\n")
            rawstring_resolution = data[-1].rstrip().lstrip().split()
            self._extcamimg_width = int(rawstring_resolution[0])
            self._extcamimg_height = int(rawstring_resolution[1])

    def triggerframe(self, request, context):
        self._pcins.captureframe()
        self._width = self._pcins.getframewidth()
        self._height = self._pcins.getframeheight()
        self._hasframe = True

        return pxmsg.Empty()

    def gettextureimg(self, request, context):
        """
        get texture image as an array

        :return: a Height*Width*1 np array,
        author: weiwei
        date: 20180207
        """

        if self._hasframe:
            textureraw = self._pcins.gettexture()
            texturearray = np.array(textureraw).reshape((self._height, self._width))
            return pxmsg.CamImg(width=self._width, height=self._height, channel=1,
                                image=np.ndarray.tobytes(texturearray))
        else:
            raise ValueError("Trigger a frame first!")

    def getdepthimg(self, request, context):
        """
        get depth image as an array

        :return: a depthHeight*depthWidth*1 np array
        author: weiwei
        date: 20180207
        """

        depthmapraw = self._pcins.getdepthmap()
        deptharray = np.array(depthmapraw).reshape((self._height, self._width))
        return pxmsg.CamImg(width=self._width, height=self._height, channel=1, image=np.ndarray.tobytes(deptharray))

    def getpcd(self, request, context):
        """
        get the full poind cloud of a new frame as an array

        :param mat_tw yaml string storing mat_tw
        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20181121
        """
        pointcloudraw = self._pcins.getpointcloud()
        pcdarray = np.array(pointcloudraw).reshape((-1, 3))
        return pxmsg.PointCloud(points=np.ndarray.tobytes(pcdarray))

    def getnormals(self, request, context):
        """
        get the normals of each pixel
        normals share the same datastructure as pointcloud
        the return value is therefore pointcloud

        :param request:
        :param context:
        :return:
        """

        normalsraw = self._pcins.getnormals()
        normalsarray = np.array(normalsraw).reshape((-1, 3))
        return pxmsg.PointCloud(points=np.ndarray.tobytes(normalsarray))

    def getrgbtextureimg(self, request, context):
        """
        get the rgb texture as an array
        author: hao chen
        date: 20220318
        :return:
        """
        rgbtextureraw = self._pcins.getrgbtexture()
        rgbtexturearray = np.array(rgbtextureraw)
        return pxmsg.CamImg(width=self._width, height=self._height, channel=3,
                            image=np.ndarray.tobytes(rgbtexturearray))

    def getextcamimg(self, request, context):
        extcamimgraw = self._pcins.getexternalcamimg()
        extcamimgarray = np.array(extcamimgraw).reshape((self._extcamimg_height, self._extcamimg_width, 3)).astype(
            float)
        return pxmsg.CamImg(width=self._extcamimg_width, height=self._extcamimg_height, channel=3,
                            image=np.ndarray.tobytes(extcamimgarray))

    def changeledpower(self, request, context):
        ledpower = request.ledpower
        if 0 <= ledpower <= 4095:
            self._pcins.changeledpower(ledpower)
        return pxmsg.Empty()


def serve(serialno="2019-09-051-LC3", host="127.0.0.1:18300"):
    portno = 65499
    resolution = "high"
    calibpath = "D:\chen\phoxi_server_tst\calib_external_cam_custom\calibration.txt"
    calibpath = ""
    try:
        pcins = pctrl.PhoxiControl(serialno, portno, resolution, calibpath)
    except:
        pcins = pctrl.PhoxiControl(serialno, portno, resolution)
    pcins.get_intrinsic_mat()
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    pxserver = PhoxiServer()
    pxserver.initialize(pcins, calibpath)
    pxrpc.add_PhoxiServicer_to_server(pxserver, server)
    server.add_insecure_port(host)
    server.start()
    print("The Phoxi server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve(serialno="2019-04-009-LC3", host="127.0.0.1:18300")
