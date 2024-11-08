# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: remote_exe.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='remote_exe.proto',
  package='RemoteExecuter',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10remote_exe.proto\x12\x0eRemoteExecuter\"\x07\n\x05\x45mpty\"\x1b\n\x0bMotionBatch\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"_\n\x06Status\x12\x31\n\x05value\x18\x01 \x01(\x0e\x32\".RemoteExecuter.Status.StatusValue\"\"\n\x0bStatusValue\x12\t\n\x05\x45RROR\x10\x00\x12\x08\n\x04\x44ONE\x10\x01\x32U\n\x0eRemoteExecuter\x12\x43\n\nrun_motion\x12\x1b.RemoteExecuter.MotionBatch\x1a\x16.RemoteExecuter.Status\"\x00\x62\x06proto3'
)



_STATUS_STATUSVALUE = _descriptor.EnumDescriptor(
  name='StatusValue',
  full_name='RemoteExecuter.Status.StatusValue',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DONE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=135,
  serialized_end=169,
)
_sym_db.RegisterEnumDescriptor(_STATUS_STATUSVALUE)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='RemoteExecuter.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=43,
)


_MOTIONBATCH = _descriptor.Descriptor(
  name='MotionBatch',
  full_name='RemoteExecuter.MotionBatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='RemoteExecuter.MotionBatch.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=72,
)


_STATUS = _descriptor.Descriptor(
  name='Status',
  full_name='RemoteExecuter.Status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='RemoteExecuter.Status.value', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _STATUS_STATUSVALUE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=74,
  serialized_end=169,
)

_STATUS.fields_by_name['value'].enum_type = _STATUS_STATUSVALUE
_STATUS_STATUSVALUE.containing_type = _STATUS
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['MotionBatch'] = _MOTIONBATCH
DESCRIPTOR.message_types_by_name['Status'] = _STATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'remote_exe_pb2'
  # @@protoc_insertion_point(class_scope:RemoteExecuter.Empty)
  })
_sym_db.RegisterMessage(Empty)

MotionBatch = _reflection.GeneratedProtocolMessageType('MotionBatch', (_message.Message,), {
  'DESCRIPTOR' : _MOTIONBATCH,
  '__module__' : 'remote_exe_pb2'
  # @@protoc_insertion_point(class_scope:RemoteExecuter.MotionBatch)
  })
_sym_db.RegisterMessage(MotionBatch)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'remote_exe_pb2'
  # @@protoc_insertion_point(class_scope:RemoteExecuter.Status)
  })
_sym_db.RegisterMessage(Status)



_REMOTEEXECUTER = _descriptor.ServiceDescriptor(
  name='RemoteExecuter',
  full_name='RemoteExecuter.RemoteExecuter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=171,
  serialized_end=256,
  methods=[
  _descriptor.MethodDescriptor(
    name='run_motion',
    full_name='RemoteExecuter.RemoteExecuter.run_motion',
    index=0,
    containing_service=None,
    input_type=_MOTIONBATCH,
    output_type=_STATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_REMOTEEXECUTER)

DESCRIPTOR.services_by_name['RemoteExecuter'] = _REMOTEEXECUTER

# @@protoc_insertion_point(module_scope)
