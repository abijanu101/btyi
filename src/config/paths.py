import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
SRC_PATH = ROOT_PATH.joinpath('src')

CONF_PATH = SRC_PATH.joinpath('config')
CORE_PATH = SRC_PATH.joinpath('core')
UI_PATH = SRC_PATH.joinpath('ui')

VIEWS_PATH = UI_PATH.joinpath('views')
WK_CONTENTS_PATH = VIEWS_PATH.joinpath('week_content')
