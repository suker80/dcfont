from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def image_load(path):

    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((224,224))
    img = np.expand_dims(np.asarray(img, np.float32), 2) / 255.0

    return img

def make_batch(reference_font='reference',train_root='dataset',batch_size=16,num_class=20):
    x=[]
    y=[]
    labels=[]

    fonts = [os.path.join(train_root,font) for font in os.listdir(train_root)]
    chars = os.listdir(fonts[0])
    ref_path = [os.path.join(reference_font,char) for char in chars]

    for i in range(batch_size):
        idx = np.random.choice(len(chars))
        random_font_idx = np.random.choice(len(fonts))
        label = random_font_idx
        labels.append(np.eye(num_class)[label])
        img_path = ref_path[idx]
        ref_img  = image_load(img_path)

        target_path = os.path.join(fonts[random_font_idx],chars[idx])
        target_image = image_load(target_path)

        x.append(ref_img)
        y.append(target_image)
    return x,y,labels


def make_batch2(reference_font='reference',target='calli3',batch_size=16,num_class=20):
    x=[]
    y=[]
    labels=[]

    chars = os.listdir(target)
    charss = [char.split('.')[0] for char in os.listdir(reference_font)]

    ref_path = [os.path.join(reference_font,char) for char in chars]
    fonts = [os.path.join(target,font) for font in os.listdir(target)]

    for i in range(batch_size):
        idx = np.random.choice(len(chars))
        random_font_idx = 5
        label = np.zeros(shape=[20])
        labels.append(label)
        img_path = ref_path[idx]
        ref_img  = image_load(img_path)

        target_path = os.path.join(target,chars[idx])
        target_image = image_load(target_path)

        x.append(ref_img)
        y.append(target_image)
    return x,y,labels
make_batch()