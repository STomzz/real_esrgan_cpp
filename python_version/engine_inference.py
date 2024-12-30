import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 告诉flake8忽略“imported but unused”错误
import numpy as np
import cv2


# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    """Load a TensorRT engine from a file."""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    """
    Allocate input/output buffers and CUDA streams for TensorRT inference.
    Returns host buffers, device buffers, and CUDA stream.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    Perform inference using the TensorRT engine.
    """
    # Transfer input data to the GPU
    for inp in inputs:
        cuda.memcpy_htod(inp['device'], inp['host'])

    # Execute inference
    context.execute_v2(bindings=bindings)

    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh(out['host'], out['device'])

    return [out['host'] for out in outputs]


def preprocess_image(image_path):
    """
    Preprocess the input image to match the model's requirements.
    Args:
        image_path: Path to the input image.
        input_shape: Model input shape as (C, H, W).

    Returns:
        Preprocessed image as a numpy array.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return img.ravel()


def postprocess_image(output_data):
    """
    Postprocess the output image to match the original image format.
    Args:
        img: Output image as a numpy array.

    Returns:
        Postprocessed image as a numpy array.
    """
    output_data = output_data.reshape(3, 1024, 2048)
    output_data = np.transpose(output_data[[2, 1, 0], :, :], (1, 2, 0))
    output = (output_data * 255.0).round().astype(np.uint8)

    return output


# Main logic
if __name__ == "__main__":
    # Load the TensorRT engine
    engine_path = "/root/zst/Realesrgan/esrgan_cpp/engine/realesrgan-x4_2.engine"
    engine = load_engine(engine_path)

    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Create TensorRT execution context
    with engine.create_execution_context() as context:
        # Prepare input data (example: random data)
        input_data = preprocess_image('/root/zst/Realesrgan/esrgan_cpp/data/00003.png',)
        np.copyto(inputs[0]['host'], input_data.ravel())

        # Run inference
        output_data = do_inference(context, bindings, inputs, outputs, stream)[0]
        output = postprocess_image(output_data)
        cv2.imwrite("/root/zst/Realesrgan/esrgan_cpp/results/output_pycuda.png", output)

        # Print results
        print("Output process successfully!")
