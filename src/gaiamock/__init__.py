'''
gaiamock: simulate Gaia epoch astrometry and fit astrometric binary orbits.

All of the functions in gaiamock.py are available directly as gaiamock.<function>.
'''
from importlib.metadata import version

from .gaiamock import *  # noqa: F401,F403
from .data import data_path, download_data, get_data_dir  # noqa: F401
from .dr4 import list_dr4_sources, load_dr4_epoch_astrometry  # noqa: F401

__version__ = version('gaiamock')
