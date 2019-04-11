import os
import tqdm
import numpy as np
import pickle
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    keys = list(dict.keys())
    for k in keys:
        new_k = k.decode(encoding='utf-8')
        dict[new_k] = dict.pop(k)
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(folder):
    tr_data = np.empty((0, 32*32*3))
    tr_labels = np.empty(1)
    '''
    32x32x3
    '''
    for i in range(1, 6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            tr_data = data_dict['data']
            tr_labels = data_dict['labels']
        else:
            tr_data = np.vstack((tr_data, data_dict['data']))
            tr_labels = np.hstack((tr_labels, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    te_data = data_dict['data']
    te_labels = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    label_names = [n.decode(encoding='utf-8') for n in bm['label_names']]
    return tr_data, tr_labels, te_data, te_labels, label_names


def save_images(images, labels, files, names, out_dir):
    for img, lbl, f in zip(tqdm.tqdm(images), labels, files):
        target_name = os.path.join(out_dir, names[lbl], f)
        img = np.reshape(img, (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        pil_img = Image.fromarray(img)
        pil_img.save(target_name)

if __name__ == '__main__':
    datapath = "./cifar-10-batches-py"
    tr_data10, tr_labels10, te_data10, te_labels10, label_names10 = \
        get_cifar10(datapath)

    tr_files10 = ['{0:06d}.png'.format(i) for i in range(len(tr_data10))]
    te_files10 = ['{0:06d}.png'.format(i) for i in range(len(te_data10))]

    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    for name in label_names10:
        os.makedirs(os.path.join('train', name), exist_ok=True)
        os.makedirs(os.path.join('test', name), exist_ok=True)

    save_images(tr_data10, tr_labels10, tr_files10, label_names10, 'train')
    save_images(te_data10, te_labels10, te_files10, label_names10, 'test')
