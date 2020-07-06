import random
import warnings
from skimage import transform
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
from config import configs
from PIL import Image
class MyDataset():
    def __init__(self, train, path, split=0.8, transform=None):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        self.data_x, self.data_y = read_fits(path=path, split_train=split, train=train)
        self.transform = transform

        np.random.seed(123456)
        np.random.shuffle(self.data_x)
        np.random.seed(123456)
        np.random.shuffle(self.data_y)

        length = len(self.data_x)

        self.data_x = np.array(self.data_x, dtype='uint8')
        self.data_y = np.array(self.data_y)

        print("========================")
        print(self.data_x.shape)
        print(self.data_y.shape)

        # self.train_x = torch.from_numpy(self.train_x).float()
        self.data_y = torch.from_numpy(self.data_y).long()

    def __getitem__(self, index):
        if self.transform is not None:
            im = Image.fromarray(self.data_x[index])
            x = self.transform(im)
            return x, self.data_y[index]
        else:
            return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.data_x.shape[0]


def get_file_name(elem):
    return elem[1]


def search_fits(path):
    fits_list = []
    path_list = []
    label_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            temp_list = []
            if os.path.splitext(file)[1] == '.fits':
                temp_list.append(root)
                temp_list.append(file)
                temp_list.append(root.split("\\")[-1])
                fits_list.append(temp_list[:])
                fits_list.sort(key=get_file_name)

    for item in fits_list:
        path_list.append(os.path.join(item[0], item[1]))
        label_list.append(item[2])

    return path_list, label_list


def label2num(label):
    if label.endswith('alpha'):
        print("alpha")
        return 0
    elif label.endswith('beta'):
        print("beta")
        return 1
    elif label.endswith('betax'):
        print("betax")
        return 2
    else:
        print(label)
        return -1


def others2uint8(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_grey = 255 * (data - data_min) / (data_max - data_min)
    data_grey = np.array(data_grey, dtype=np.uint8)


def read_fits(path, train, channels=2, split_train=0.8):
    from astropy.io import fits
    path_list, label_list = search_fits(path)
    np.random.seed(233333)
    np.random.shuffle(path_list)
    np.random.seed(233333)
    np.random.shuffle(label_list)
    image_list = []
    label_list_2channels = []
    count = 0
    split_length = int(split_train * (len(path_list) / 2))

    if channels == 1:
        if train:
            for i in range(0, split_length * 2):
                # count += 1
                # print(count)
                hdul = fits.open(path_list[i])
                hdul.verify('silentfix')
                scaled_img = transform.resize(hdul[1].data, (224, 224))
                image_list.append(others2uint8(scaled_img))
                hdul.close()

        else:
            for i in range(split_length * 2, len(path_list)):
                # count += 1
                # print(count)
                hdul = fits.open(path_list[i])
                hdul.verify('silentfix')
                scaled_img = transform.resize(hdul[1].data, (224, 224))
                image_list.append(others2uint8(scaled_img))
                hdul.close()

    elif channels == 2:
        if train:
            for i in range(0, split_length * 2, 2):
                count += 1
                print(count)
                temp_list = []
                for j in range(2):
                    if path_list[i+j].find(label_list[i+j]) == -1:
                        print("Error!!!")
                        print(label_list[i+j])
                    hdul = fits.open(path_list[i+j])
                    hdul.verify('silentfix')
                    scaled_img = transform.resize(hdul[1].data, (224, 224))
                    temp_list.append(scaled_img)
                    hdul.close()
                image_list.append(temp_list[:])

                label_list_2channels.append(label_list[i])
        else:
            for i in range(split_length * 2, len(path_list), 2):
                count += 1
                print(count)
                temp_list = []
                # print(label_list[i])
                for j in range(2):
                    # print(path_list[i+j])
                    hdul = fits.open(path_list[i+j])
                    hdul.verify('silentfix')
                    scaled_img = transform.resize(hdul[1].data, (224, 224))
                    temp_list.append(scaled_img)
                    hdul.close()
                image_list.append(temp_list[:])
                label_list_2channels.append(label_list[i])
    else:
        pass

    if channels == 2:
        label_num_list = np.array([label2num(label) for label in label_list_2channels])
    else:
        label_num_list = np.array([label2num(label) for label in label_list])

    image_list = np.array(image_list)
    image_list = image_list.transpose((0, 2, 3, 1))

    print(image_list.shape)
    print(label_num_list.shape)
    return image_list, label_num_list


def load_data():
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          #transforms.RandomRotation(90),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5], [0.5, 0.5])])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5], [0.5, 0.5])])

    train_data = MyDataset(path="data/trainset", train=True, split=0.8, transform=transform_train)
    val_data = MyDataset(path="data/trainset", train=False, split=0.8, transform=transform_val)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=configs.bs, shuffle=True, num_workers=configs.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=configs.bs, shuffle=False, num_workers=configs.workers, pin_memory=True)

    return train_loader, val_loader


