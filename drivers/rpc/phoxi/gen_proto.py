from grpc_tools import protoc

protoc.main("grpc_tools.protoc -I=./ --python_out=. --grpc_python_out=. ./phoxi.proto".split())