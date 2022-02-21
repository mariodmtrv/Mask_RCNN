import os
from os import walk
import shutil

images_relative_path = "images"
labels_relative_path = "labels"
instances_relative_path = "instances"
panoptic_relative_path = "panoptic"

root_path = '/home/mario/mapilary-vistas/mapilary'
target_root_path = '/home/mario/mapilary-vistas/mapillary_25'


def make_dirs(set_type):
    os.makedirs("{}/{}/{}".format(target_root_path, set_type, images_relative_path), exist_ok=True)
    os.makedirs("{}/{}/{}".format(target_root_path, set_type, labels_relative_path), exist_ok=True)
    os.makedirs("{}/{}/{}".format(target_root_path, set_type, instances_relative_path), exist_ok=True)
    os.makedirs("{}/{}/{}".format(target_root_path, set_type, panoptic_relative_path), exist_ok=True)


def handle_test_set(target_count):
    os.makedirs("{}/{}/{}".format(target_root_path, 'testing', images_relative_path), exist_ok=True)
    target_image_ids = pick_instance_subset(target_count, 'testing')
    for image_id in target_image_ids:
        image_path = "{}/{}/{}.jpg".format('testing', images_relative_path, image_id)
        shutil.copyfile("{}/{}".format(root_path, image_path),
                        "{}/{}".format(target_root_path, image_path))


def pick_instance_subset(target_count, set_type):
    image_files = []
    for (dirpath, dirnames, filenames) in walk("{}/{}/{}".format(root_path, set_type, images_relative_path)):
        image_files.extend(filenames)
        break
    target_image_ids = list(map(lambda name: name[:-4], image_files[:target_count]))
    print(target_image_ids)
    return target_image_ids


def main():
    os.makedirs(target_root_path, exist_ok=True)
    for set_type in ['training', 'validation']:
        make_dirs(set_type)
    target_count = 25
    for set_type in ['training', 'validation']:
        handle_set(set_type, target_count)
    handle_test_set(target_count)


def handle_set(set_type, target_count):
    target_image_ids = pick_instance_subset(target_count, set_type)
    for image_id in target_image_ids:
        image_path = "{}/{}/{}.jpg".format(set_type, images_relative_path, image_id)
        label_path = "{}/{}/{}.png".format(set_type, labels_relative_path, image_id)
        instance_path = "{}/{}/{}.png".format(set_type, instances_relative_path, image_id)
        panoptic_path = "{}/{}/{}.png".format(set_type, panoptic_relative_path, image_id)
        shutil.copyfile("{}/{}".format(root_path, image_path),
                        "{}/{}".format(target_root_path, image_path))
        shutil.copyfile("{}/{}".format(root_path, label_path),
                        "{}/{}".format(target_root_path, label_path))
        shutil.copyfile("{}/{}".format(root_path, instance_path),
                        "{}/{}".format(target_root_path, instance_path))
        shutil.copyfile("{}/{}".format(root_path, panoptic_path),
                        "{}/{}".format(target_root_path, panoptic_path))


if __name__ == "__main__":
    main()
