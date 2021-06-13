# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: outputTrans.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='outputTrans.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x11outputTrans.proto\"\x0c\n\noutputData\"\x1a\n\tinputData\x12\r\n\x05idata\x18\x01 \x03(\x02\x32\x32\n\x05Trans\x12)\n\x0coutput_trans\x12\x0b.outputData\x1a\n.inputData\"\x00\x62\x06proto3'
)




_OUTPUTDATA = _descriptor.Descriptor(
  name='outputData',
  full_name='outputData',
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
  serialized_start=21,
  serialized_end=33,
)


_INPUTDATA = _descriptor.Descriptor(
  name='inputData',
  full_name='inputData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='idata', full_name='inputData.idata', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=35,
  serialized_end=61,
)

DESCRIPTOR.message_types_by_name['outputData'] = _OUTPUTDATA
DESCRIPTOR.message_types_by_name['inputData'] = _INPUTDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

outputData = _reflection.GeneratedProtocolMessageType('outputData', (_message.Message,), {
  'DESCRIPTOR' : _OUTPUTDATA,
  '__module__' : 'outputTrans_pb2'
  # @@protoc_insertion_point(class_scope:outputData)
  })
_sym_db.RegisterMessage(outputData)

inputData = _reflection.GeneratedProtocolMessageType('inputData', (_message.Message,), {
  'DESCRIPTOR' : _INPUTDATA,
  '__module__' : 'outputTrans_pb2'
  # @@protoc_insertion_point(class_scope:inputData)
  })
_sym_db.RegisterMessage(inputData)



_TRANS = _descriptor.ServiceDescriptor(
  name='Trans',
  full_name='Trans',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=63,
  serialized_end=113,
  methods=[
  _descriptor.MethodDescriptor(
    name='output_trans',
    full_name='Trans.output_trans',
    index=0,
    containing_service=None,
    input_type=_OUTPUTDATA,
    output_type=_INPUTDATA,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_TRANS)

DESCRIPTOR.services_by_name['Trans'] = _TRANS

# @@protoc_insertion_point(module_scope)
