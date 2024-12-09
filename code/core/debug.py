import os
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

def output_debug_image(img, logger, name, dated=True):
    '''
    params
    img: np.array
    name: str (should contain extension)
    dated: bool (whether to prefix the filename with the date and time)
    '''
    try:
        final_img = np.array(img, dtype=np.uint8)
        if final_img.shape[0] < 5: # this is the color dimension
            final_img = np.transpose(final_img, [1, 2, 0])
        # maybe add a date to the name to keep it from being overwritten between images
        if dated:
            datestring = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f'{datestring}_{os.path.basename(name)}'
            name = os.path.join(os.path.dirname(name), filename)
        plt.imsave(name, final_img)
        logger.info(f'------------- print image ---------------')
        logger.info(f'shape: {img.shape}')
        logger.info(f'range: {np.min(final_img)} to {np.max(final_img)}')
        logger.info(f'save as: {name}')
    except Exception as e:
        logger.info(f'ERROR: unable to save image {name} with shape {img.shape}')
        logger.info(e)