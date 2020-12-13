import torch
import torchvision.transforms as transforms
import numpy as np
from wct import PhotoWCT
from PIL import Image
'''
A stylizer program which performs neural
style transfer using a content image
and a style image.
'''

class Stylizer:

    '''
    Initializes the stylizer which runs on 
    device DEVICE.
    '''
    def __init__(self, device : str):
        self.device = device
        self.model = PhotoWCT()
        self.model.load_state_dict(torch.load("../model.pth")) # TODO: remove hardcode
        self.model.to(self.device)
        self.cont_img = None
        self.styl_img = None
    
    '''
    Image loader function
    '''
    def __load(self, img):
        if type(img) == str:
            img = Image.open(img).convert('RGB')
            img = transforms.ToTensor()(img)
        if isinstance(img, np.ndarray):
            # unsqueeze 2d array
            img = torch.from_numpy(img)
        if isinstance(img, torch.Tensor):
            if len(img.shape) == 2:
                img = img.repeat(3, 1, 1)
            # convert HWC to CHW shape
            elif img.shape[2] == 1 or img.shape[2] == 3:
                img = img.permute(2, 0, 1)
        else:
            raise ValueError("Image type not understood")
        # add a batch size dimension to BCHW shape
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return img

    '''
    Reads a content image given the path string
    or Numpy array or Pytorch tensor.
    '''
    def load_content(self, cont_img):
        self.cont_img = self.__load(cont_img)
        
    '''
    Reads a style image given the path string
    or Numpy array or Pytorch tensor.
    '''
    def load_style(self, styl_img):
        self.styl_img = self.__load(styl_img)

    '''
    Runs the stylizer to produce the 
    style transferred result image.
    '''
    def run(self):
        with torch.no_grad():
            result = self.model.transform(self.cont_img, self.styl_img).squeeze(0).cpu()
        # must make contiguous to be sent through MPI
        result = result.permute(1, 2, 0).contiguous().numpy()
        return result

