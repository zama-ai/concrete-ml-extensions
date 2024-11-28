#!/bin/bash

python -c "import concrete_ml_extensions as fhext; print('CUDA enabled: ', fhext.is_cuda_enabled());"

python -c "import concrete_ml_extensions as fhext; print('CUDA available: ', fhext.is_cuda_available());"
