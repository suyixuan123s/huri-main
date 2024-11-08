# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import huri.test.app.remote_exe_pb2 as remote__exe__pb2


class RemoteExecuterStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.run_motion = channel.unary_unary(
                '/RemoteExecuter.RemoteExecuter/run_motion',
                request_serializer=remote__exe__pb2.MotionBatch.SerializeToString,
                response_deserializer=remote__exe__pb2.Status.FromString,
                )


class RemoteExecuterServicer(object):
    """Missing associated documentation comment in .proto file."""

    def run_motion(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RemoteExecuterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'run_motion': grpc.unary_unary_rpc_method_handler(
                    servicer.run_motion,
                    request_deserializer=remote__exe__pb2.MotionBatch.FromString,
                    response_serializer=remote__exe__pb2.Status.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'RemoteExecuter.RemoteExecuter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RemoteExecuter(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def run_motion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/RemoteExecuter.RemoteExecuter/run_motion',
            remote__exe__pb2.MotionBatch.SerializeToString,
            remote__exe__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
