from os import rename, listdir, makedirs
from os.path import join

# Amount of files to be used for training
split = 0.8


def create_class(class_name, dataset_base='PetImages', destination_base='data'):
    '''
    Move the files of one class to their new location
    :param class_name: Name of the class
    :param dataset_base: The path to the Microsoft dataset
    :param destination_base: The path, where the dataset should be stored
    '''
    # get all files of the given class
    files = listdir(join(dataset_base, class_name))
    data_split = round(len(files)*split)

    # split the images
    train_files = files[:data_split]
    test_files = files[data_split:]

    # create the needed dirs
    makedirs(join(destination_base, 'train', class_name))
    makedirs(join(destination_base, 'test', class_name))

    # move the files
    for file in train_files:
        rename(join(dataset_base, class_name, file), join(destination_base, 'train', class_name, file))

    for file in test_files:
        rename(join(dataset_base, class_name, file), join(destination_base, 'test', class_name, file))


def prepare_dataset_structure(dataset_base='PetImages', destination_base='data'):
    '''
    Moves the files from the Microsoft dataset to the structure we need.
    :param dataset_base: The path to the Microsoft dataset
    :param destination_base: The path, where the dataset should be stored
    '''
    create_class('Cat', dataset_base=dataset_base, destination_base=destination_base)
    create_class('Dog', dataset_base=dataset_base, destination_base=destination_base)


if __name__ == '__main__':
    prepare_dataset_structure()
