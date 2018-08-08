import numpy as np
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from directed_exploration.logging_ops import init_logging
import logging

logger = logging.getLogger(__name__)


def generate_boxpush_heatmap_from_npy_records(directory, file_prefix, delete_records=False):

    file_names = [file_name for file_name in os.listdir(directory)
                  if file_name.endswith(".npy") and file_name.startswith(file_prefix)]

    location_records = np.concatenate([np.load(os.path.join(directory, file_name)) for file_name in file_names], axis=0)

    max_heatmap_samples = 10000

    location_records = location_records[np.random.choice(len(location_records),
                                                         min(max_heatmap_samples, len(location_records)),
                                                         replace=False)]

    # Add a minuscule amount of variation in case agent doesn't move on a certain axis.
    location_records = location_records + (np.random.randn(*location_records.shape) / 1000)

    location_records = location_records.swapaxes(0, 1)

    z = gaussian_kde(location_records)(location_records)
    idx = z.argsort()
    x, y, z = location_records[0, idx], location_records[1, idx], z[idx]

    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, c=z, s=80, edgecolors='',  cmap=plt.cm.jet, alpha=0.7)
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    im = plt.imread(os.path.join(directory, 'level.png'))
    plt.imshow(im, extent=[0, 100, 0, 100], aspect='auto')
    plt.axis('equal')
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    heatmap_image_path = os.path.join(directory, "{}_heatmap.png".format(file_prefix))
    plt.savefig(heatmap_image_path, transparent=True, bbox_inches='tight', pad_inches=0)

    if delete_records:
        for file_name in file_names:
            try:
                os.remove(os.path.join(directory, file_name))
            except OSError:
                pass

    return heatmap_image_path


if __name__ == '__main__':
    init_logging()

    generate_boxpush_heatmap_from_npy_records('itexplore_20180525145205/heatmap_records', 'it1', delete_records=False)
