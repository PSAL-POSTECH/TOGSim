"""A layer on top of TOGSim: ACS writes the cycle and unit tables of the kernels in kernels/ from formulas, and
the ONNX frontend maps a graph's nodes to those kernels and replays each one. The simulator does not depend on it."""
