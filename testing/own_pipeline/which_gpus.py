import torch

if __name__ == "__main__":
    try:
        print('___CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        if torch.cuda.is_available():
            print('__CUDA Device Name:', torch.cuda.get_device_name(0))
            print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory/1e9)
        else:
            print('CUDA is not available. Please check your NVIDIA driver and CUDA installation.')
    except Exception as e:
        print(f'Error: {e}')
