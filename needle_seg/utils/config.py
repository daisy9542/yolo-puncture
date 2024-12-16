import yaml
import os

__all__ = ['get_config']


class Dict2Obj:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = Dict2Obj(value)
            self.__dict__[key] = value


dir_path = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(dir_path, '../../config.yaml')
# print(config_file_path)

# yaml配置文件
with open(config_file_path, 'r') as f:
    yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
config = Dict2Obj(yaml_cfg)


def get_config():
    return config
