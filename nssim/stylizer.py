import torch
import torchvision.transforms as transforms
import numpy as np
from .wct import PhotoWCT
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
    def __init__(self, device : str, size: int):
        self.device = device
        self.model = PhotoWCT()
        self.model.load_state_dict(torch.load("nssim/model.pth"))
        self.model.to(self.device)
        self.cont_img = None
        self.styl_img = None
        self.size = size
    
    '''
    Image loader function
    '''
    def __load(self, img):
        if type(img) == str:
            img = Image.open(img).convert('RGB')
            # resize if necessary
            img = img.resize((self.size, self.size))
            img = transforms.ToTensor()(img)
            
        if isinstance(img, np.ndarray):
            # unsqueeze 2d array
            img = torch.from_numpy(img)
        if isinstance(img, torch.Tensor):
            if len(img.shape) == 2: # HW -> NCHW
                img = img.repeat(3, 1, 1)
                img = img.unsqueeze(0)
            elif len(img.shape) == 3: 
                if img.shape[2] == 1 or img.shape[2] == 3: # HWC->CHW
                    img = img.permute(2, 0, 1)
                    img = img.unsqueeze(0) #CHW -> NCHW
                elif img.shape[0] != 3:
                    img = img.unsqueeze(1) # NHW -> NCHW
                    img = img.repeat(1, 3, 1, 1) 
                else:
                    img = img.unsqueeze(0)
            elif img.shape[3] == 1 or img.shape[3] == 3: # NWHC ->NCHW
                img = img.permute(0, 3, 1, 2)

        else:
            raise ValueError("Image type not understood")
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
            result = self.model.transform(self.cont_img, self.styl_img).cpu()
        # must make contiguous to be sent through MPI
        result = result.permute(0, 2, 3, 1).contiguous().numpy()
        return result

