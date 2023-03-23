# Map Annotation

API for processing and serving map elements stored in [GeoPackage](https://en.wikipedia.org/wiki/GeoPackage) (*.gpkg) format. 

## Setup

### Installation
This repository can be installed as a pip package. Navigate to the root of the repository (the location of the `pyproject.toml` file) and enter the following command:
```bash
$ pip install .
```

If the source code of this project will be edited frequently, it may convenient to install the package in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) instead:
```bash
$ pip install -e .
```
This will save you from having to reinstall the package after every change in the source code. Do note the limitations of development before using.

Be aware that installing pip packages with environment managers like `conda` may cause dependency conflicts or unexpected behaviour.

### Data
This repository includes annotation data for the locations that were recorded in the [View-of-Delft dataset](https://github.com/tudelft-iv/view-of-delft-dataset). The data can be found in the `data` directory. 'Raw' annotations are in the `data/annotations/` directory, with processed data in the `data/processed` directory.

Raw annotations - if formatted as detailed in the following section - can be processed to the format expected by the functions and classes in this package using the `process.py` script. The syntax is:
```bash
$ python process.py -i path/to/raw/annotations -o path/to/processed/annotations
```

E.g. for the structure of the `data` directory of this repository:
```bash
$ python process.py -i data/annotations -o data/processed
```
**Note: the `process.py` script currently does not work.**

#### Annotation format
TODO: explainer on the format of the annotations i.e. what fields are expected for each map element.

## Using the API
The API consists of a number of classes and functions that make working with common map elements more convenient. 

TODO: write documentation.

