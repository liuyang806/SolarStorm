import torch
import os
from PIL import Image
from torchvision import transforms
from skimage import transform
import torch.backends.cudnn as cudnn
from config import configs
import torch.nn as nn
import numpy as np
def read_fits(path):
    from astropy.io import fits
    image_list = []
    count = 0
    
    c_path = path + 'continuum/'
    m_path = path + 'magnetogram/'
    image_adds = os.listdir(c_path)
    for image in image_adds:
        count += 1
        #print(count)
        temp_list = []
        image = image.strip()
        image_name = os.path.splitext(image)[0]
        #print(image_name[:-10])
        image_file_c = c_path + '/' + image
        image_file_m = m_path + '/' + image_name[:-10] + '.magnetogram.fits'

        hdul1 = fits.open(image_file_c)
        hdul1.verify('silentfix')
        scaled_img_c = transform.resize(hdul1[1].data, (224, 224))
        temp_list.append(scaled_img_c)
        hdul1.close()

        hdul2 = fits.open(image_file_m)
        hdul2.verify('silentfix')
        scaled_img_m = transform.resize(hdul2[1].data, (224, 224))
        temp_list.append(scaled_img_m)
        hdul2.close()

        image_list.append(temp_list[:])

    image_list = np.array(image_list)
    image_list = image_list.transpose((0, 2, 3, 1))

    print(image_list.shape)
    return image_list, count


classes = ['alpha','beta','betax']
transforms=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5], [0.5, 0.5])])
cpk_filename = configs.checkpoints + os.sep + configs.model_name + "-checkpoint.pth.tar"
best_cpk = cpk_filename.replace("-checkpoint.pth.tar","-best_model.pth.tar")
checkpoint = torch.load(best_cpk)
cudnn.benchmark = True
net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
net.conv1 = nn.Sequential(
          nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1,bias=False),
          nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True),
          nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True),
          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )
net.cuda()
net.load_state_dict(checkpoint['state_dict'])
net.eval()
def prediect(img_path):
    with torch.no_grad():
        test_list, num = read_fits(img_path)
        #img=Image.open(img_path).convert('RGB')
        #img=transforms(img).unsqueeze(0)
        for i in range(num):
            test_list = np.array(test_list, dtype='uint8')
            im = Image.fromarray(test_list[i])
            img = transforms(im).unsqueeze(0)
            img_ = img.cuda()
            outputs = net(img_)
            _, predicted = torch.max(outputs, 1)
            print('this picture maybe :',predicted[0])#classes[int(str(predicted[0])[7])])

if __name__ == '__main__':
    prediect('./test/')
    #read_fits('./test/')
