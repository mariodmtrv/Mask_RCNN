import os
from os import walk
import shutil
images_relative_path = "training/images"
labels_relative_path = "training/labels"
instances_relative_path = "training/instances"
panoptic_relative_path = "training/panoptic"

root_path = '/home/mario/mapilary-vistas/mapilary/'
target_root_path = '/home/mario/mapilary-vistas/mapillary_25'

def make_dirs():
    os.makedirs(target_root_path, exist_ok=True)
    os.makedirs("{}/{}".format(target_root_path, images_relative_path), exist_ok=True)
    os.makedirs("{}/{}".format(target_root_path, labels_relative_path), exist_ok=True)
    os.makedirs("{}/{}".format(target_root_path, instances_relative_path), exist_ok=True)
    os.makedirs("{}/{}".format(target_root_path, panoptic_relative_path), exist_ok=True)

def main():
    make_dirs()
    target_count = 25
    image_files = []
    for (dirpath, dirnames, filenames) in walk("{}/{}".format(root_path, images_relative_path)):
        image_files.extend(filenames)
        break
    target_image_ids = list(map(lambda name: name[:-4], image_files[:target_count]))
    print(target_image_ids)

    for image_id in target_image_ids:
        image_path = "{}/{}.jpg".format(images_relative_path, image_id)
        label_path = "{}/{}.png".format(labels_relative_path, image_id)
        instance_path = "{}/{}.png".format(instances_relative_path, image_id)
        panoptic_path = "{}/{}.png".format(panoptic_relative_path, image_id)
        shutil.copyfile("{}/{}".format(root_path, image_path), "{}/{}".format(target_root_path, image_path))
        shutil.copyfile("{}/{}".format(root_path, label_path), "{}/{}".format(target_root_path, label_path))
        shutil.copyfile("{}/{}".format(root_path, instance_path), "{}/{}".format(target_root_path, instance_path))
        shutil.copyfile("{}/{}".format(root_path, panoptic_path), "{}/{}".format(target_root_path, panoptic_path))


if __name__ == "__main__":
    main()