import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')
CONF_PATH = SRC_PATH.joinpath('config')

# datasets
DATA_PATH = ROOT_PATH.joinpath('data')
XSUM_PATH = DATA_PATH.joinpath('xsum')

# models
MODELS_PATH = ROOT_PATH.joinpath('models')
PRETRAINED_PATH = MODELS_PATH.joinpath('pretrained')
TRAINED_PATH = MODELS_PATH.joinpath('trained')
TUNED_PATH = MODELS_PATH.joinpath('tuned')


# streamlit
UI_PATH = SRC_PATH.joinpath('ui')
VIEWS_PATH = UI_PATH.joinpath('views')
WK_CONTENTS_PATH = VIEWS_PATH.joinpath('week_content')
