import torch
import torch.nn as nn
import glob
import os
import sys
import numpy as np 
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def get_images(paths, labels, nb_sample=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """

    if nb_sample is not None:
        sampler = lambda x: random.sample(x, nb_sample)
    else:
        sampler = lambda x: x
    
    labels_images = [(i, os.path.join(path, image))
                    for i, path in zip(labels, paths)
                    for image in sampler(os.listdir(path))]
    
    if shuffle:
        random.shuffle(labels_images)
    
    return labels_images

def image_file_to_array(filename, dim_input):

    image = np.asarray(plt.imread(filename))
    image = np.reshape(image, [dim_input])
    image = 1. - image.astype(np.float32) / 255.

    return image

class DataGenerator(object):

    def __init__(self, num_classes, num_samples_per_class, config={}):

        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        
        data_folder = config.get('data_folder', './omniglot_resized')

        self.image_size = config.get('image_size', (28, 28))
        self.input_dim = np.prod(self.image_size)
        self.output_dim = self.num_classes

        character_folder = [os.path.join(data_folder, alphabet, character)
                            for alphabet in os.listdir(data_folder)
                            if os.path.isdir(os.path.join(data_folder, alphabet))
                            for character in os.listdir(os.path.join(data_folder, alphabet))
                            if os.path.isdir(os.path.join(data_folder, alphabet, character))]

        random.seed(1)
        random.shuffle(character_folder)

        meta_train = 1100 
        meta_val = 100

        self.meta_train_characters = character_folder[: meta_train]
        self.meta_val_characters = character_folder[meta_train: meta_train+meta_val]
        self.meta_test_characters = character_folder[meta_train+meta_val: ]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """

        if batch_type == "train":
            folders = self.meta_train_characters
        if batch_type == "test":
            folders = self.meta_test_characters
        if batch_type == "val":
            folders = self.meta_val_characters

        num_batches = len(folders)//batch_size
        folders = folders[:num_batches*batch_size]
        all_image_batches = []
        all_label_batches = []

        for batch_idx in range(batch_size):
            sample_classes =  random.sample(folders, self.num_classes)
            #sample_classes = folders[batch_idx*self.num_classes : (batch_idx+1)*self.num_classes]
            one_hot_labels = np.identity(self.num_classes)

            labels_images = get_images(sample_classes, one_hot_labels, nb_sample=self.num_samples_per_class, shuffle=False)
            train_images = []
            train_labels = []    
            for sample_idx, (labels, images) in enumerate(labels_images):
                train_images.append(image_file_to_array(images, 784))
                train_labels.append(labels)

            
            train_images, train_labels = shuffle(train_images, train_labels)

            labels = np.vstack(train_labels).reshape((-1, self.num_classes, self.num_classes))  # K, N, N
            images = np.vstack(train_images).reshape((self.num_samples_per_class, self.num_classes, -1))  # K x N x 784

            all_label_batches.append(labels)
            all_image_batches.append(images)

        all_image_batches = np.stack(all_image_batches).astype(np.float32)
        all_label_batches = np.stack(all_label_batches).astype(np.float32)

        return all_label_batches, all_image_batches      
