#!/usr/bin/env python3
import os
import time
import logging
import tensorrt as trt
import numpy as np
from cuda import cudart
import ctypes
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DETECTOR_KEY = "objectdetection"

def cuda_call(call):
    err, res = call[0], call[1:]
    if err != 0:
        raise RuntimeError(f"CUDA call failed with error code {err}")
    if len(res) == 1:
        return res[0]
    return res

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype, shape: Tuple[int, ...]):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        self.shape = shape
        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
    @property
    def host(self) -> np.ndarray:
        return self._host
    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        # Flatten and copy only as much as fits
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')
    @property
    def device(self) -> int:
        return self._device
    @property
    def nbytes(self) -> int:
        return self._nbytes
    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"
    def __repr__(self):
        return self.__str__()
    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

class TensorRtDetector:
    type_key = DETECTOR_KEY

    def __init__(self, detector):
        trt.init_libnvinfer_plugins(None, "")
        trt_path = detector

        # Determine if the path is an ONNX or TRT engine file
        is_onnx = trt_path.endswith('.onnx')
        is_trt = trt_path.endswith('.trt')

        self.trt_logger = trt.Logger()

        # If ONNX, always build engine and save as TRT
        if is_onnx:
            print("Building engine from ONNX:", trt_path)
            engine = self.build_engine_from_onnx(trt_path, fp16=True)
            if engine is None:
                raise RuntimeError(f"Failed to build TensorRT engine from ONNX file: {trt_path}")
            # Save engine to .trt file for future use
            trt_save_path = trt_path.replace('.onnx', '.trt')
            with open(trt_save_path, "wb") as fw:
                fw.write(engine.serialize())
            print("Saved engine to", trt_save_path)
            self.engine = engine
        elif is_trt and os.path.exists(trt_path):
            print("Reading TensorRT engine from", trt_path)
            with open(trt_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine from file: {trt_path}")
        else:
            raise FileNotFoundError(f"Model file not found or unsupported extension: {trt_path}")

        # Defensive: check engine is not None
        if self.engine is None:
            raise RuntimeError("TensorRT engine is None after initialization.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def allocate_buffers(self, engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            if profile_idx is None:
                shape = engine.get_binding_shape(binding)
            else:
                shape = engine.get_profile_shape(profile_idx, binding)[2]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, but no profile was specified.")
            size = trt.volume(shape)
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
            bindingMemory = HostDeviceMem(size, dtype, shape)
            bindings.append(int(bindingMemory.device))
            if engine.binding_is_input(binding):
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream

    def build_engine_from_onnx(self, onnx_file_path, fp16=False):
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.trt_logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             builder.create_builder_config() as config, \
             trt.OnnxParser(network, self.trt_logger) as parser, \
             trt.Runtime(self.trt_logger) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2Gb
            if fp16:
                print("setting fp16 flag")
                config.set_flag(trt.BuilderFlag.FP16)
            if not os.path.exists(onnx_file_path):
                print(f'Cannot find ONNX file: {onnx_file_path}')
                return None
            with open(onnx_file_path, 'rb') as fr:
                if not parser.parse(fr.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                print("ERROR: Failed to build serialized TensorRT network from ONNX.")
                return None
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                print("ERROR: Failed to deserialize TensorRT engine from serialized plan.")
                return None
            return engine

    def infer(self, image):
        # Defensive: check input shape matches expected
        input_arr = np.array(image)
        expected_shape = self.inputs[0].host.shape
        if input_arr.size > self.inputs[0].host.size:
            raise ValueError(
                f"Tried to fit an array of size {input_arr.size} into host memory of size {self.inputs[0].host.size}"
            )
        # Flatten and copy only as much as fits
        np.copyto(self.inputs[0].host[:input_arr.size], input_arr.flat, casting='safe')
        host2device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        for inp in self.inputs:
            cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, host2device, self.stream))
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        device2host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for out in self.outputs:
            cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, device2host, self.stream))
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return [out.host for out in self.outputs]

    def detect_raw(self, tensor_input):
        input_arr = np.array(tensor_input)
        # Defensive: check input shape matches expected
        if input_arr.shape != tuple(self.inputs[0].shape):
            # Try to reshape if possible, else raise
            try:
                input_arr = input_arr.reshape(self.inputs[0].shape)
            except Exception as e:
                raise ValueError(f"Input shape {input_arr.shape} does not match expected {self.inputs[0].shape}") from e
        trt_outputs = self.infer(image=input_arr)
        # Defensive: check output shapes
        if len(trt_outputs) < 4:
            raise RuntimeError("TensorRT model did not return enough outputs")
        num_dets = min(100, len(trt_outputs[1]) // 4, len(trt_outputs[2]), len(trt_outputs[3]))
        det_boxes = []
        h = input_arr.shape[2]
        w = input_arr.shape[3]
        for box in range(num_dets):
            x0, y0, x1, y1 = map(round, trt_outputs[1][box*4:(box+1)*4])
            det_boxes.append((x0*1.0/w, y0*1.0/h, x1*1.0/w, y1*1.0/h))
        det_scores = trt_outputs[2][:num_dets]
        det_classes = trt_outputs[3][:num_dets]

        detections = np.zeros((20, 8), np.float32)
        det_idx = 0
        for i in range(num_dets):
            if det_idx == 20:
                break
            if float(det_scores[i]) < 0.1:
                continue
            detections[det_idx] = [
                det_classes[i],
                float(det_scores[i]),
                det_boxes[i][0],
                det_boxes[i][1],
                det_boxes[i][2],
                det_boxes[i][3],
                0.0,
                0.0
            ]
            det_idx += 1
        return detections
