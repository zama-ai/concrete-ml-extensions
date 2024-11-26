__name__ = "concrete-ml-extensions"
__author__ = "Zama"
__all__ = ["concrete-ml-extensions"]
__version__ = "0.1.3"

import .concrete_ml_extensions as backend

def default_params():
    return backend.default_params()

def is_cuda_available():
    return backend.is_cuda_available()


