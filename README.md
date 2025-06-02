# FourCastNet Hurricane Intensity Estimation

This repository contains code and resources for estimating hurricane intensity using FourCastNet, a deep learning model trained on atmospheric reanalysis data. The project supports both ERA5 and MERRA reanalysis datasets.

## Project Structure

```
.
├── example/                      # Example code and notebooks
│   ├── run_inference.ipynb      # Jupyter notebook for running inference
│   └── src.py                   # Source code for the model
├── files/                       # Sample data files
│   ├── ERA_*.nc                # ERA5 reanalysis data
│   ├── MERRA_*.nc              # MERRA reanalysis data
│   ├── *_ERA.csv               # ERA5 intensity data
│   ├── *_MERRA.csv             # MERRA intensity data
│   └── *_HURDAT.csv            # HURDAT2 intensity data
├── ERA_scaling/                 # ERA5 data scaling parameters
├── MERRA_scaling/              # MERRA data scaling parameters
├── bias_correction_checkpoints_ERA/    # ERA5 model checkpoints
└── bias_correction_checkpoints_MERRA/  # MERRA model checkpoints
```

## Features

- Hurricane intensity estimation using FourCastNet
- Support for both ERA5 and MERRA reanalysis datasets
- Bias correction for improved accuracy
- Example notebooks for running inference
- Pre-trained model checkpoints

## Requirements

The project requires Python 3.8+ and the following key dependencies:

### Core Scientific Computing
- NumPy (>=1.21.0): For numerical computations and array operations
- SciPy (>=1.7.0): For scientific computing and signal processing
- Pandas (>=1.3.0): For data manipulation and analysis

### Deep Learning
- PyTorch (>=1.9.0): Deep learning framework
- PyTorch Geometric (>=2.0.0): For graph neural networks

### Data Processing
- Xarray (>=0.19.0): For handling labeled multi-dimensional arrays
- netCDF4 (>=1.5.7): For reading/writing NetCDF files

### Machine Learning
- scikit-learn (>=0.24.0): For machine learning utilities and metrics

### Visualization
- Matplotlib (>=3.4.0): For plotting and visualization

### Development Tools
- tqdm (>=4.62.0): For progress bars
- glob2 (>=0.7): For file path operations
- Jupyter (>=1.0.0): For interactive development
- Notebook (>=6.4.0): For running Jupyter notebooks
- ipykernel (>=6.0.0): For Jupyter kernel support

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:
```bash
git clone [repository-url]
cd FourCastNet_Hurricane_Intensity_Estimation
```

2. Install dependencies (requirements.txt to be added)

3. Run the example notebook:
```bash
jupyter notebook example/run_inference.ipynb
```

## Data

The project uses two main reanalysis datasets:
- ERA5: European Centre for Medium-Range Weather Forecasts (ECMWF) reanalysis
- MERRA: Modern-Era Retrospective Analysis for Research and Applications

Sample data files are provided in the `files/` directory for testing and demonstration purposes.

## Model Checkpoints

Pre-trained model checkpoints are available in:
- `bias_correction_checkpoints_ERA/` for ERA5 models
- `bias_correction_checkpoints_MERRA/` for MERRA models

## License

This project is licensed under the terms of the included LICENSE file.

## Citation

If you use this code in your research, please cite:
[Citation information to be added]

## Contact

[Contact information to be added]