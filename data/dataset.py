import json
import random

import pandas as pd
import torch
import os
import numpy as np
import re
import pickle as pkl
from PIL import Image
import h5py
import torch.utils.data as data
from collections import Counter
from torchvision import transforms
from itertools import groupby
import warnings
import csv
import pandas

print(os.getcwd())

root_data = './data/datasets/'
train_sentences_path = os.path.join(root_data, 'sentences_train.json')
test_sentences_path = os.path.join(root_data, 'sentences_test.json')
val_sentences_path = os.path.join(root_data, 'sentences_val.json')
batch_size = 128
data_workers = 8


def get_loader(mode, fix_length=20, max_detections=20):
    """ Returns a data loader for the desired split """
    if mode == 'train':
        sentences_path = train_sentences_path
    elif mode == 'val':
        sentences_path = val_sentences_path
    elif mode == 'test':
        sentences_path = test_sentences_path
    split = COCOSetField(
        sentences_path, classes_path="./data/object_class_list.txt", features_file="./data/genome-trainval.h5", precomp_glove_path="./data/object_class_glove.pkl", fix_length=fix_length, max_detections=max_detections
    )

    loader = torch.utils.data.DataLoader(
        split,
        batch_size=batch_size,
        shuffle=True if mode == 'train' else False,
        pin_memory=True,
        num_workers=data_workers
    )

    return loader



class COCOSetField(data.Dataset):
    """COCO seq Dataset"""
    def __init__(self, sentences_path=None, classes_path=None, features_file=None, precomp_glove_path=None, fix_length=20, max_detections=20):
        self.fix_len = fix_length
        self.max_detections = max_detections
        self.classes = []
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(sentences_path, 'r') as fd:
            self.sentences_pair = json.load(fd)

        # self.sentences_pair = pd.read_csv(sentences_path)

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        # visual
        self.image_feature_path = features_file
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.image_ids = [sen['image_id'] for sen in self.sentences_pair]


    def _create_coco_id_to_index(self):
        with h5py.File(self.image_feature_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_ids_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_ids_to_index

    def _load_image(self, image_id):
        """ Load an image"""
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.image_feature_path, 'r')

        coco_ids = self.features_file['ids'][()]

        coco_ids = list(coco_ids)
        image_id = int(image_id)

        if image_id in coco_ids:
            index = self.coco_id_to_index[image_id]
            img = self.features_file['features'][index]
            boxes = self.features_file['boxes'][index]
            width = self.features_file['widths'][index]
            height = self.features_file['heights'][index]
            object_ids = self.features_file['objects_id'][index]

            return torch.from_numpy(img).transpose(0, 1), torch.from_numpy(boxes).transpose(0, 1), width, height, object_ids


    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou


    def preprocess(self, x):
        id_image = x["image_id"]
        det_classes = x["inter_entities1"]  # gt
        try:
            det_features, det_boxes, width, height, object_ids = self._load_image(id_image)
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        selected_classes = x["inter_entities2"]  # gt

        # input
        # det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = torch.zeros((self.fix_len, det_features.shape[-1]))
        det_sequences_word = torch.zeros((self.fix_len, 300))
        det_sequences_position = torch.zeros((self.fix_len, 4))

        # gt
        gt_sequences_visual = torch.zeros((self.fix_len, det_features.shape[-1]))
        gt_sequences_word = torch.zeros((self.fix_len, 300))
        gt_sequences_position = torch.zeros((self.fix_len, 4))

        # gt: [1, 2, 3, 4, 5]
        # input: [1, 3, 5]
        # cls_seq = det_classes[:self.fix_length]
        # cls_seq.sort()

        det_classes = det_classes[:self.fix_len]
        selected_classes = selected_classes[:self.fix_len]

        hard_perm_matrix = torch.zeros((self.fix_len, self.fix_len))

        det_ind = torch.full((self.fix_len, 1), -1, dtype=torch.float)
        gt_ind = torch.full((self.fix_len, 1), -1, dtype=torch.float)


        for j, (det_class, gt_class) in enumerate(zip(det_classes, selected_classes)):
            if gt_class in det_classes:
                hard_perm_matrix[j][det_classes.index(gt_class)] = 1
            if det_class in self.vectors:
                det_sequences_word[j] = torch.from_numpy(self.vectors[det_class])
            if det_class not in self.classes:
                continue
            object_id = self.classes.index(det_class)
            det_ind[j][0] = object_id
            det_ids = [i for i, j in enumerate(object_ids) if j == object_id]
            if len(det_ids) == 0:
                continue
            det_sequences_visual[j] = det_features[det_ids[0]]
            bbox = det_boxes[det_ids[0]]
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height


            if gt_class in self.vectors:
                gt_sequences_word[j] = torch.from_numpy(self.vectors[gt_class])

            if gt_class not in self.classes:
                continue
            object_id = self.classes.index(gt_class)
            gt_ind[j][0] = object_id
            gt_ids = [i for i, j in enumerate(object_ids) if j == object_id]
            if len(gt_ids) == 0:
                continue
            gt_sequences_visual[j] = det_features[gt_ids[0]]
            bbox = det_boxes[gt_ids[0]]
            gt_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
            gt_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
            gt_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
            gt_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height

        return det_sequences_word, det_sequences_visual, \
            det_sequences_position, gt_sequences_word, \
            gt_sequences_visual, gt_sequences_position, hard_perm_matrix, det_ind, gt_ind

    def __getitem__(self, item):
        x = self.sentences_pair[item]
        return self.preprocess(x)


    def __len__(self):
        return len(self.sentences_pair)


class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)