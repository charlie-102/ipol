"""
iio shim module - wraps imageio.v3 to provide iio-compatible API.

Install by copying to site-packages or adding to PYTHONPATH:
    cp iio_shim.py $(python -c "import site; print(site.getsitepackages()[0])")/iio.py

Or install as a module:
    conda activate ipol_kervrann
    cp iio_shim.py "$(python -c 'import site; print(site.getsitepackages()[0])')/iio.py"
"""

import imageio.v3 as _iio

version = "shim-1.0"

def read(filename):
    """Read an image file."""
    return _iio.imread(filename)

def write(filename, data):
    """Write an image to file."""
    return _iio.imwrite(filename, data)
