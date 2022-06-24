from samples.balloon.balloon import BalloonConfig, BalloonDataset
from samples.mapillary.mapillary import MapillaryDataset, MapillaryConfig
import  mrcnn.utils
import mrcnn.visualize as visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import numpy as np

def test_balloon():
    dataset = BalloonDataset()
    DATASET = "/home/mario/Projects/Thesis/balloon-dataset/balloon"
    dataset.load_balloon(DATASET, 'train')
    dataset.prepare()
    image_ids = np.random.choice(dataset.image_ids, 4)
    print(image_ids)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
