#!/usr/bin/env python3
import os
import time
import logging
import argparse
import tensorrt as trt
import numpy as np
from cuda import cudart
import ctypes
from typing import Optional, List, Tuple
import cv2

try:
    import experiments.model.OurModel.detector as trt
    from cuda import cuda
    TRT_SUPPORT = True
except ModuleNotFoundError as e:
    TRT_SUPPORT = False

from typing import Literal

logger = logging.getLogger(__name__)

DETECTOR_KEY = "segmentor"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


def cuda_call(call):
    err, res = call[0], call[1:]
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype, shape: Tuple[int, int, int]):
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


class TensorRtSegmentor:
    type_key = DETECTOR_KEY

    trt_logger = trt.Logger()

    def __init__(self, detector):
        trt.init_libnvinfer_plugins(None, "")
        trt_path = detector
        onnx_path = trt_path.replace('.trt', '.onnx')
        self.engine = None

        # Always build engine from ONNX if .trt is missing or invalid
        need_build = False
        if not os.path.exists(trt_path):
            logger.warning(f"TensorRT engine file {trt_path} does not exist. Will build from ONNX.")
            need_build = True
        else:
            # Try to load the engine, catch deserialization errors
            try:
                logger.info(f"Reading TensorRT engine from {trt_path}")
                with open(trt_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                if self.engine is None:
                    logger.error(f"Deserialized engine is None. Will rebuild from ONNX.")
                    need_build = True
            except Exception as e:
                logger.error(f"Failed to load TensorRT engine from {trt_path}: {e}")
                need_build = True

        if need_build:
            logger.info(f"Building TensorRT engine from ONNX: {onnx_path}")
            self.engine = self.build_engine_from_onnx(onnx_path, fp16=True)
            if self.engine is not None:
                try:
                    with open(trt_path, "wb") as fw:
                        fw.write(self.engine.serialize())
                    logger.info(f"Saved engine to {trt_path}")
                except Exception as e:
                    logger.error(f"Failed to save engine to {trt_path}: {e}")
            else:
                raise RuntimeError(f"Failed to build TensorRT engine from ONNX: {onnx_path}")

        if self.engine is None:
            raise RuntimeError(f"TensorRT engine could not be loaded or built for {trt_path}")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    def allocate_buffers(self, engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [str(i) for i in engine]
        for binding in tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = engine.get_binding_shape(binding) if profile_idx is None else engine.get_binding_profile_shape(binding, profile_idx)[-1]
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
            if not engine.binding_is_input(binding):
                dtype = np.dtype(np.float32)
            # Allocate host and device buffers
            bindingMemory = HostDeviceMem(size, dtype, shape)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(bindingMemory)
            else:
                outputs.append(bindingMemory)
        return inputs, outputs, bindings, stream

    def build_engine_from_onnx(self, onnx_file_path, fp16=False):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.trt_logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             builder.create_builder_config() as config, \
             trt.OnnxParser(network, self.trt_logger) as parser, \
             trt.Runtime(self.trt_logger) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2Gb
            if fp16:
                logger.info("Setting FP16 flag for TensorRT engine build")
                config.set_flag(trt.BuilderFlag.FP16)
            # Parse model file
            if not os.path.exists(onnx_file_path):
                logger.error(f"Cannot find ONNX file: {onnx_file_path}")
                return None
            with open(onnx_file_path, 'rb') as fr:
                if not parser.parse(fr.read()):
                    logger.error('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                logger.error("Failed to build serialized network from ONNX.")
                return None
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                logger.error("Failed to deserialize CUDA engine from plan.")
            return engine

    def infer(self, image):
        self.inputs[0].host = image
        # Transfer input data to the GPU.
        host2device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, host2device, self.stream)) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        # Transfer predictions back from the GPU.
        device2host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, device2host, self.stream)) for out in self.outputs]
        # Synchronize the stream
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        # Return outputs
        return [out.host for out in self.outputs]

    def detect_raw(self, tensor_input):
        # tensor_input is expected to be (1, 3, H, W)
        # Remove batch dimension if present
        if tensor_input.ndim == 4 and tensor_input.shape[0] == 1:
            image_array = tensor_input[0]
        else:
            image_array = tensor_input

        # image_array is now (3, H, W), convert to (H, W, 3)
        if image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0

        # Resize to 320x320
        image_resized = cv2.resize(image_array, (320, 320), interpolation=cv2.INTER_LINEAR)

        # Convert back to (1, 3, 320, 320)
        image_resized = np.transpose(image_resized, (2, 0, 1))[np.newaxis, ...]

        input = self._scale_image(
            image_array=image_resized,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        trt_outputs = self.infer(image=input)
        # trt_outputs[0] is a flat array of shape (102400,), need to reshape to (320, 320)
        mask = trt_outputs[0]
        if mask.ndim == 1 and mask.size == 320 * 320:
            mask = mask.reshape((320, 320))
        else:
            mask = np.squeeze(mask)
            if mask.shape != (320, 320):
                raise ValueError(f"Unexpected mask shape after squeeze: {mask.shape}")
        # Resize mask back to input size (H, W)
        h, w = tensor_input.shape[2], tensor_input.shape[3]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.round(mask)
        return mask
    
    def _scale_image(self,image_array, mean, std):
        """
        Scale an RGB image represented as a NumPy array with mean and standard deviation.

        Args:
            image_array (numpy.ndarray): Input image represented as a NumPy array with shape (height, width, channels).
            mean (tuple): Mean values for each channel (R, G, B).
            std (tuple): Standard deviation values for each channel (R, G, B).

        Returns:
            numpy.ndarray: Scaled image represented as a NumPy array.
        """
        # Convert the image array to float32 (if not already)
        image_array = image_array.astype(np.float32)

        # Scale the image
        scaled_image = (image_array - np.array(mean).reshape(1, -1, 1, 1)) / np.array(std).reshape(1, -1, 1, 1)

        return scaled_image.astype(np.float32)