"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from .models import VGGEncoder, VGGDecoder


class PhotoWCT(nn.Module):
    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)

    def transform_once(self, cont_img, styl_img):
        sF2, sF1 = self.e2.forward_multiple(styl_img)

        cF2, cpool_idx, cpool = self.e2(cont_img)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1)
        Im1 = self.d1(csF1)
        return Im1


    
    def transform(self, cont_img, styl_img):

        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        #sF4 = sF4.data.squeeze(0)
        #cF4 = cF4.data.squeeze(0)
        csF4 = self.__feature_wct(cF4, sF4)
        
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        #sF3 = sF3.data.squeeze(0)
        #cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        #sF2 = sF2.data.squeeze(0)
        #cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        #sF1 = sF1.data.squeeze(0)
        #cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1)
        Im1 = self.d1(csF1)
        return Im1

    def __feature_wct(self, cont_feat, styl_feat):
        cont_b, cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2), cont_feat.size(3)
        styl_b, styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2), styl_feat.size(3)
        cont_feat_view = cont_feat.view(cont_b, cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_b, styl_c, -1).clone()

        target_feature = self.__wct_core(cont_feat_view, styl_feat_view)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float()
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 2)  # b x c x (h x w)
        c_mean = c_mean.unsqueeze(2).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[1]).unsqueeze(0)  # .double()
        if self.is_cuda:
            iden = iden.cuda()
        
        contentConv = torch.bmm(cont_feat, cont_feat.permute(0, 2, 1)).div(cFSize[2] - 1) + iden
        #del iden
        
        c_u, c_e, c_v = torch.svd(contentConv, some=True)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        
        #k_c = cFSize[1]
        #for i in range(cFSize[1] - 1, -1, -1):
        #    if c_e[i] >= 0.00001:
        #        k_c = i + 1
        #        break
        k_c = torch.max(torch.sum(c_e >= 0.00001, 1)) + 1
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 2) 
        styl_feat = styl_feat - s_mean.unsqueeze(2).expand_as(styl_feat)
        styleConv = torch.bmm(styl_feat, styl_feat.permute(0, 2, 1)).div(sFSize[2] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=True)

        # broadcast style image if the batch size of style feature is 1
        if styl_feat.size(0) == 1:
            s_u = s_u.repeat(cont_feat.size(0), 1, 1)
            s_e = s_e.repeat(cont_feat.size(0), 1)
            s_v = s_v.repeat(cont_feat.size(0), 1, 1)
        
        #k_s = sFSize[0]
        #for i in range(sFSize[0] - 1, -1, -1):
        #    if s_e[i] >= 0.00001:
        #        k_s = i + 1
        #        break
        k_s = torch.max(torch.sum(c_e >= 0.00001, 1)) + 1
        
        c_d = (c_e[:, 0:k_c]).pow(-0.5)
        step1 = torch.bmm(c_v[:, :, 0:k_c], torch.diag_embed(c_d))
        step2 = torch.bmm(step1, c_v[:, :, 0:k_c].permute(0, 2, 1))
        whiten_cF = torch.bmm(step2, cont_feat)

        s_d = (s_e[:, 0:k_s]).pow(0.5)
        targetFeature = torch.bmm(
            torch.bmm(
                torch.bmm(s_v[:, :, 0:k_s], torch.diag_embed(s_d)), 
                s_v[:, :, 0:k_s].permute(0, 2, 1)),
                whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(2).expand_as(targetFeature)
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass