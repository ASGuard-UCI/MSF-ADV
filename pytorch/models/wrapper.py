from models import Render4CNN
from utils import registry

import config

@registry.register('Network', 'Render4CNN')
class Render4CNNWrapper(Render4CNN):
    '''
    Similar to Render4CNN, but load weights automatically from config instead of
    specifying in the arguments.
    '''

    def __init__(self, finetune=False, weights=None, num_classes=12):
        if weights == 'lua':
            super().__init__(finetune=finetune, weights=weights,
                weights_path=config.RENDER4CNN_WEIGHTS, num_classes=num_classes)

        elif weights == 'npy':
            raise NotImplementedError


    def prediction_from_output(self, output):
        '''
        Compute the network predictions from the model output.
        This is because sometimes a network may have multiple supervisions
        but we only need the predictions to evaluate.
        '''
        # TODO: figure out
        pass
