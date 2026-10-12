"""`python -m togsim_frontend.onnx_frontend <model.onnx> --config <yml>`."""
from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
