"""Convert Compositor's pinned DepthMini Core ML graph to ONNX for CPU inference.

Pass the official model.mlmodel and weight.bin, and a coremltools 9.0 wheel for
its protobuf schemas. The native coremltools runtime is unnecessary.
"""

import argparse
import hashlib
import math
import struct
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Unrecognized DepthMini asset: {path}')
    return data


def read_model(data, wheel):
    with tempfile.TemporaryDirectory() as temporary, zipfile.ZipFile(wheel) as archive:
        schemas = Path(temporary) / 'depthmini_schema'
        schemas.mkdir()
        for name in archive.namelist():
            if name.startswith('coremltools/proto/') and name.endswith('.py'):
                (schemas / PurePosixPath(name).name).write_bytes(archive.read(name))
        sys.path.insert(0, temporary)
        try:
            from depthmini_schema import Model_pb2
            return Model_pb2.Model.FromString(data)
        finally:
            sys.path.pop(0)


def tensor_value(value, blob):
    tensor = value.type.tensorType
    shape = tuple(dimension.constant.size for dimension in tensor.dimensions)
    length = math.prod(shape)
    if value.WhichOneof('value') == 'blobFileValue':
        location = value.blobFileValue
        if location.fileName != '@model_path/weights/weight.bin':
            raise ValueError(f'Unexpected weights reference: {location.fileName}')
        magic, storage_type, size, offset = struct.unpack_from('<IIQQ', blob, location.offset)
        if magic != 0xDEADBEEF or offset + size > len(blob):
            raise ValueError('Invalid weight storage record')
        raw = blob[offset:offset + size]
        if storage_type == 1 and size == length * 2:
            values = np.frombuffer(raw, '<f2').astype(np.float32)
        elif storage_type == 13 and size == (length * 6 + 7) // 8:
            bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')[:length * 6]
            values = bits.reshape(-1, 6).astype(np.uint8) @ (1 << np.arange(6, dtype=np.uint8))
        else:
            raise ValueError(f'Unsupported weight storage type {storage_type}')
    else:
        immediate = value.immediateValue.tensor
        field = immediate.WhichOneof('value')
        raw = getattr(immediate, field).values
        dtype = {'floats': np.float32, 'ints': np.int64, 'bools': np.bool_, 'strings': str}
        values = np.frombuffer(raw, '<f2').astype(np.float32) if field == 'bytes' else np.asarray(raw, dtype=dtype[field])
    return values.reshape(shape)


def convert(model, blob):
    block = model.mlProgram.functions['main'].block_specializations['CoreML8']
    constants, initializers, nodes = {}, {}, []

    for op in block.operations:
        name = op.outputs[0].name

        def names(key):
            return [argument.name for argument in op.inputs[key].arguments]

        def inputs(*keys):
            return [names(key)[0] for key in keys]

        def constant(key):
            return constants[names(key)[0]]

        def add(kind, arguments, **attributes):
            for argument in arguments:
                if argument in constants and argument not in initializers:
                    initializers[argument] = numpy_helper.from_array(constants[argument], argument)
            nodes.append(helper.make_node(kind, arguments, [name], name=name, **attributes))

        kind = op.type
        if kind == 'const':
            constants[name] = tensor_value(op.attributes['val'], blob)
        elif kind == 'constexpr_lut_to_dense':
            indices = tensor_value(op.inputs['indices'].arguments[0].value, blob)
            palette = tensor_value(op.inputs['lut'].arguments[0].value, blob)
            if palette.shape != (1, 1, 1, 1, 64, 1):
                raise ValueError('Unexpected palette shape')
            constants[name] = palette.flatten()[indices]
        elif kind == 'cast':
            if constant('dtype').item() != 'fp16':
                raise ValueError('Unexpected depth model cast')
            # CPU kernels use float32. Values of the half-precision weights and
            # palettes remain exact after expansion.
            add('Identity', inputs('x'))
        elif kind in ('add', 'sub', 'mul'):
            add(kind.capitalize(), inputs('x', 'y'))
        elif kind == 'conv':
            top, bottom, left, right = constant('pad').tolist()
            if constant('pad_type').item() not in ('custom', 'valid'):
                raise ValueError('Unsupported convolution padding')
            add('Conv', inputs('x', 'weight', 'bias'), pads=[top, left, bottom, right],
                strides=constant('strides').tolist(), dilations=constant('dilations').tolist(), group=int(constant('groups')))
        elif kind == 'relu':
            add('Relu', inputs('x'))
        elif kind == 'thresholded_relu':
            add('ThresholdedRelu', inputs('x'), alpha=float(constant('alpha')))
        elif kind == 'sigmoid_hard':
            add('HardSigmoid', inputs('x'), alpha=float(constant('alpha')), beta=float(constant('beta')))
        elif kind == 'reduce_mean':
            add('ReduceMean', inputs('x'), axes=constant('axes').tolist(), keepdims=int(constant('keep_dims')))
        elif kind == 'upsample_bilinear':
            scale_name = name + '_scales'
            constants[scale_name] = np.asarray([1, 1, constant('scale_factor_height'), constant('scale_factor_width')], dtype=np.float32)
            mode = 'align_corners' if constant('align_corners') else 'half_pixel'
            add('Resize', [names('x')[0], '', scale_name], mode='linear', coordinate_transformation_mode=mode)
        elif kind == 'concat':
            if constant('interleave'):
                raise ValueError('Unsupported interleaved concatenation')
            add('Concat', names('values'), axis=int(constant('axis')))
        else:
            raise ValueError(f'Unsupported MIL operator: {kind}')

    graph = helper.make_graph(nodes, 'DepthMini',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 'height', 'width'])],
        [helper.make_tensor_value_info('depth', TensorProto.FLOAT, [1, 1, 'height', 'width'])],
        list(initializers.values()))
    converted = helper.make_model(graph, ir_version=8, producer_name='Compositor Windows', opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(converted, full_check=True)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_directory', type=Path)
    parser.add_argument('schema_wheel', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    graph = verified(args.model_directory / 'model.mlmodel', 'c9e2f4f1e44409a2b9f5fb0b072ba41fb42d03a555c333f635f5162676feb744')
    weights = verified(args.model_directory / 'weight.bin', '63d066f1aefc362f2cc0805a624e08df13b6cff3be45dc4307b20a531a2a9f86')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(convert(read_model(graph, args.schema_wheel), weights), args.output)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
