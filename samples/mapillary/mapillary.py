import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import warnings
from numpy import zeros, newaxis
from PIL import Image
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MapillaryConfig(Config):
    """Configuration for training on the Mapillary dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mapillary"

    # We use a GPU with 12GB memory, which can fit two images.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 66  # Background + categories from the config
    NUM_CLASSES = 66 # Skipping background
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 25

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    SELECTED_LABELS = {"animal--bird",
                       "human--person",
                       "human--rider--bicyclist",
                       "human--rider--motorcyclist",
                       "object--bench",
                       "object--vehicle--car",
                       "object--fire-hydrant",
                       "object--traffic-light",
                       "object--vehicle--bus",
                       "object--vehicle--motorcycle",
                       "object--vehicle--truck",
                       "background"}


############################################################
#  Dataset
############################################################

class MapillaryDataset(utils.Dataset):
    dataset_config: MapillaryConfig

    def set_config(self, dataset_config: MapillaryConfig):
        self.dataset_config = dataset_config

    def load_mapillary(self, dataset_dir, subset):
        """Load a subset of the Mapillary dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # read in config file
        with open('config.json') as config_file:
            config = json.load(config_file)
        # in this example we are only interested in the labels
        labels = config['labels']

        # print labels
        print("There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            if self.dataset_config.SELECTED_LABELS:
                if not label["name"] in self.dataset_config.SELECTED_LABELS:
                    continue
            self.add_class("mapillary", label_id, label["name"])
            print("{:>30} ({:2d}): {:<40} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))


        # Train or validation dataset?
        assert subset in ["train", "val"]
        subset_dir = {"train": "training", "val": "validation"}
        dataset_dir = os.path.join(dataset_dir, subset_dir[subset])
        images_dir = "{}/images".format(dataset_dir)
        for item in os.listdir(images_dir):
            if os.path.isfile(os.path.join(images_dir, item)):
                image_id = os.path.splitext(item)[0]
                image_path = "{}/images/{}.jpg".format(dataset_dir, image_id)
                instance_path = "{}/instances/{}.png".format(dataset_dir, image_id)
                instance_image = Image.open(instance_path)
                instance_array: np.array = np.array(instance_image, dtype=np.uint16)

                # now we split the instance_array into labels and instance ids
                instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
                # instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)
                classes = np.unique(instance_label_array)
                classes = classes.astype('int32')
                instance_masks = []
                for clazz in classes:
                    layer = np.zeros(instance_label_array.shape, dtype=np.bool8)
                    layer[instance_label_array == clazz] = True
                    instance_masks.append(layer)
                instances = np.stack(instance_masks, axis=2).astype(np.bool8)
                self.add_image(
                    "mapillary",
                    image_id=image_id,  # use file name as a unique image id
                    path=image_path,
                    width=instance_image.width, height=instance_image.height,
                    instance=instances,
                    classes=classes
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a mapillary dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "mapillary":
            return super(self.__class__, self).load_mask(image_id)
        return image_info['instance'], image_info['classes']

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mapillary":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MapillaryDataset()
    dataset_train.load_mapillary(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MapillaryDataset()
    dataset_val.load_mapillary(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    epochs = 5
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Mapillary concepts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/mapillary/dataset/",
                        help='Directory of the Mapillary dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MapillaryConfig()
    else:
        class InferenceConfig(MapillaryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))