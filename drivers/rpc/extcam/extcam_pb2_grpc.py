# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import drivers.rpc.extcam.extcam_pb2 as extcam__pb2


class CamStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getimg = channel.unary_unary(
                '/ExtCam.Cam/getimg',
                request_serializer=extcam__pb2.Empty.SerializeToString,
                response_deserializer=extcam__pb2.CamImg.FromString,
                )


class CamServicer(object):
    """Missing associated documentation comment in .proto file."""

    def getimg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CamServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getimg': grpc.unary_unary_rpc_method_handler(
                    servicer.get_img,
                    request_deserializer=extcam__pb2.Empty.FromString,
                    response_serializer=extcam__pb2.CamImg.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ExtCam.Cam', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Cam(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def getimg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ExtCam.Cam/getimg',
            extcam__pb2.Empty.SerializeToString,
            extcam__pb2.CamImg.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
