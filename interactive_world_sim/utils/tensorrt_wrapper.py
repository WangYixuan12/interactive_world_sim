import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class TensorRTWrapper:
    """Wrapper class for TensorRT engine to handle inference with dynamic shapes."""

    def __init__(
        self, model_path: str, state_shape: tuple, action_shape: tuple
    ) -> None:
        # Initialize CUDA context
        self.ctx = pycuda.autoinit.context

        # Load TensorRT engine
        with open(model_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Get tensor names and I/O information
        self.input_names = []
        self.output_names = []
        for i, _ in enumerate(self.engine):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Set default shapes (modify if needed for your model)
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Configure dynamic shapes
        self._configure_dynamic_shapes()

        # Allocate buffers
        self.buffers: dict[str, dict] = {}
        self.stream = cuda.Stream()
        self._allocate_buffers()

    def _configure_dynamic_shapes(self) -> None:
        """Handle dynamic input shapes if present"""
        for name in self.input_names:
            if -1 in self.engine.get_tensor_shape(name):
                if "state" in name.lower():
                    self.context.set_input_shape(name, self.state_shape)
                elif "action" in name.lower():
                    self.context.set_input_shape(name, self.action_shape)

    def _allocate_buffers(self) -> None:
        """Allocate host and device buffers for all tensors"""
        for i, _ in enumerate(self.engine):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)

            self.buffers[name] = {
                "host": np.empty(shape, dtype=dtype),
                "device": cuda.mem_alloc(np.empty(shape, dtype=dtype).nbytes),
            }

    def forward(self, state_input: np.ndarray, action_input: np.ndarray) -> dict:
        """Perform inference with the given state and action inputs

        Returns: Dictionary of output numpy arrays
        """
        # Prepare inputs
        self.buffers[self.input_names[0]]["host"][:] = state_input.astype(
            self.buffers[self.input_names[0]]["host"].dtype
        )
        self.buffers[self.input_names[1]]["host"][:] = action_input.astype(
            self.buffers[self.input_names[1]]["host"].dtype
        )

        # Transfer inputs to device
        for name in self.input_names:
            cuda.memcpy_htod_async(
                self.buffers[name]["device"], self.buffers[name]["host"], self.stream
            )

        # Set tensor addresses
        for i, _ in enumerate(self.engine):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, int(self.buffers[name]["device"]))

        # Execute inference
        self.context.execute_async_v3(self.stream.handle)

        # Transfer outputs to host
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.buffers[name]["host"], self.buffers[name]["device"], self.stream
            )

        # Synchronize stream
        self.stream.synchronize()

        # Return outputs
        return {name: self.buffers[name]["host"].copy() for name in self.output_names}

    def _copy_input_data(self, input_name: str, data: np.ndarray) -> None:
        """Copy input data to buffer with proper dtype"""
        target_dtype = self.buffers[input_name]["host"].dtype
        self.buffers[input_name]["host"][:] = data.astype(target_dtype)


# Example usage
if __name__ == "__main__":
    state_shape = (100, 1, 4, 32, 32)
    action_shape = (100, 3, 20)
    wrapper = TensorRTWrapper("dynamics_fp16.trt", state_shape, action_shape)

    for i in range(10):
        s_time = time.time()

        # Generate sample data (replace with real data)
        state = np.random.randn(state_shape).astype(np.float16)
        action = np.random.randn(action_shape).astype(np.float32)

        # Run inference
        outputs = wrapper.forward(state, action)

        # Print results
        print(f"Inference {i+1}:")
        for name, tensor in outputs.items():
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  Time: {time.time() - s_time:.4f}s\n")
