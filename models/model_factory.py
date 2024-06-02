from models.model import build_model as ganimation_modified
from models.ganimation.model import build_model as ganimation

def build_model(config):
    if config.model_name == 'ganimation':
        model = ganimation(config)
    elif config.model_name == 'ganimation_modified':
        model = ganimation_modified(config)
    else:
        raise NotImplementedError(f'Invalid model name: {config.model}')
    return model