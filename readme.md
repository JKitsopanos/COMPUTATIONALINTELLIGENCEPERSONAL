## Environment Setup
You can set up your virtual environment using either of the provided requirement files based on your hardware. 

In your virtual environment, run either `pip install -r requirements_gpu.txt` if you have a CUDA toolkit setup already and want PyTorch with CUDA support (you may have to change the "+cu121" in requirements_gpu.txt to the version your GPU is compatible with before installing), otherwise just run `pip install -r requirements.txt`.