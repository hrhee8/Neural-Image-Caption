""" This file loads datset.
    @author: Sejong
"""

import torch
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from vocabulary import Vocabulary
from pycocotools.coco import COCO


# Write a customized Dataset class compatible with torch.utils.DataLoader
class CocoDataset(Dataset):
    '''Dataset class for torch.utils.DataLoader'''
    def __init__(self, image_dir, caption_dir, vocab, transform):
        '''
        Parameters:
        ----------
            image_dir: director to coco image
            caption_dir: coco annotation json file path
            vocab: vocabulary wrapper

        '''
        self.image_dir = image_dir
        self.coco = COCO(caption_dir)
        self.keys = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, idx):
        '''
        Private function return one sample, image, caption
        '''
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')
        image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # append start and end
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return len(self.keys)

    def showImg(sef, idx):
        ''' Return a PIL Image '''
        annotation_id = self.keys[idx]
        image_id = self.coco.anns[annotation_id]['image_id']
        caption = self.coco.anns[annotation_id]['caption']
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        assert img_file_name.split('.')[-1] == 'jpg'

        image = Image.open(os.path.join(self.image_dir, img_file_name)).convert('RGB')
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption = [self.vocab('<<start>>'), *[self.vocab(x) for x in tokens], self.vocab('<<end>>')]
        target = torch.Tensor(caption)
        return image, caption


def coco_batch(coco_data):
    '''
    create mini_batch tensors from the list of tuples, this is to match the output of __getitem__()
    coco_data: list of tuples of length 2:
        coco_data[0]: image, shape of (3, 256, 256)
        coco_data[1]: caption, shape of length of the caption;
    '''
    # Sort Descending by caption length
    coco_data.sort(key=lambda x: len(x[1]), reverse = True)
    images, captions = zip(*coco_data)
    # turn images to a 4D tensor with specified batch_size
    images = torch.stack(images, 0)
    # do the same thing for caption. -> 2D tensor with equal length of max lengths in batch, padded with 0
    cap_length = [len(cap) for cap in captions]
    seq_length = max(cap_length)
    # Truncation
    if max(cap_length) > 100:
        seq_length = 100
    targets = torch.LongTensor(np.zeros((len(captions), seq_length)))
    for i, cap in enumerate(captions):
        length = cap_length[i]
        targets[i, :length] = cap[:length]

    return images, targets, cap_length


if __name__ == "__main__":
    PATH = './'
    image_dir = PATH + 'data/resized2014/'
    caption_path = PATH + 'data/annotations/captions_train2014.json'
    vocab_path = PATH + 'data/vocab.pkl'
    crop_size = 224
    batch_size = 128
    num_workers = 4

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    coco = CocoDataset(image_dir, caption_path, vocab, transform)
    dataLoader = torch.utils.data.DataLoader(coco, 128, shuffle=True, num_workers=1, collate_fn=coco_batch)
    for i, (images, captions, lengths) in enumerate(dataLoader):
        print('IMAGE:', images.shape)
        print('Captions', captions.shape)
        for i in list(captions.data.numpy()[0]):
            print(vocab.idx2word[i])
