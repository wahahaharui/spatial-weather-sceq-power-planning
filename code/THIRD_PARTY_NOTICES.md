# Third-Party Notices

This repository contains the authors' own implementation for reproducing the Lvliang city-level SCEQ power planning experiments.

## Switch Modeling Framework

The model design refers to the modeling framework and concepts of Switch for power system planning. Switch itself is not bundled in this repository as source code.

The `benders_fixed_v6` implementation is an independent implementation created by the authors based on the research model formulation and Benders decomposition workflow. It is not a direct copy of Switch source code.

Users who wish to use Switch directly should obtain it from the official Switch project and follow the license terms of that project.

## Runtime Dependencies

This project depends on third-party Python and solver packages, including but not limited to:

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- TensorFlow
- PyTorch
- PyTorch Geometric
- Gurobi / gurobipy
- GeoPandas
- Rasterio
- PyArrow
- joblib

These dependencies are not vendored in this repository. Their licenses are governed by their respective projects. A valid Gurobi installation and license are required to run the optimization workflow.

## External Data and Models

Large external model weights, calibration arrays, geospatial rasters, and generated experiment outputs are not bundled in the GitHub repository. See `data_required_manifest.csv` and `README.md` for placement instructions and publication notes.
