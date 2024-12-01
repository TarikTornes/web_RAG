import yaml, os


def load_conf():
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config
