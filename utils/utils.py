import time
from functools import wraps

__all__ = ['timeit', 'print_onnx']

# Decorator function to measure the time taken by a function
def timeit(time_len):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            wrapper.times.append(elapsed_time)
            if len(wrapper.times) % time_len == 0:
                average_time = sum(wrapper.times[-time_len:]) / time_len
                print(f"Average time for last {time_len} frames in {func.__name__}: {average_time:.4f} seconds")
            return result
        wrapper.times = []
        return wrapper
    return decorator



def print_onnx(model):
    import onnxruntime as ort

    # 加载 ONNX 模型
    session = ort.InferenceSession(model)

    # 查看模型的输入信息
    print("Model Inputs:")
    for input in session.get_inputs():
        print(f"Name: {input.name}")
        print(f"Shape: {input.shape}")
        print(f"Data Type: {input.type}")
        print("-" * 50)
        break

    # 查看模型的输出信息
    print("Model Outputs:")
    for output in session.get_outputs():
        print(f"Name: {output.name}")
        print(f"Shape: {output.shape}")
        print(f"Data Type: {output.type}")
        print("-" * 50)
        break