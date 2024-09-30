# Gaussian LERF
Code based on Splatfacto and LERF.

## File Structure

```
├── my_method
│   ├── __init__.py
│   ├── lerf_gaussian_config.py
│   ├── lerf_gaussian_pipeline.py 
│   ├── lerf_gaussian.py
│   ├── data 
│   │    ├── lerf_gaussian_datamanger.py
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "lerf_gaussian". To train with it, run the command:
```
ns-train lerf_gaussian --data [PATH]
```