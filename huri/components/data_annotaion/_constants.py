import huri.core.file_sys as fs
import logging

# logging configuration
LOGGER_LEVEL = logging.INFO
logging.basicConfig(level=LOGGER_LEVEL)

# Constants
DATA_ANNOT_PATH = fs.workdir_data.joinpath("data_annotation")
DATA_ANNOT_ONRACK_PATH = DATA_ANNOT_PATH.joinpath("tab_color_valid")
SEL_PARAM_PATH = fs.Path(__file__).parents[0].joinpath("params")
