#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""
import numpy as np

import ex2
import cv2
import utils
import json 


def checkBinarization(img, ref):
    try:
        H,W,C = img.shape
        binary = ex2.binarizeImage(img)
        #utils.show(np.equal(binary,0))
        utils.show(binary)
        utils.show(ref)
        intersection = ((ref == 255) & (binary == 255) & (ref == binary)).sum()
        union = ((ref == 255) | (binary == 255)).sum()

        iou = intersection/union
        assert binary.min() == 0, "The binary image must have values 0 or 255"
        assert binary.max() == 255, "The binary image must have values 0 or 255"
        assert iou > 0.8, "The binarization is not good enough. Try changing some parameters. Is your image maybe inverted?"
        return True

    except AssertionError as e:
        print(e)
        return False

def checkFourierDescriptor():
    try:
        points = np.random.random((20, 2))
        fd = points[:,0]+1j*points[:, 1]
        fd_ref = np.fft.fft(fd)
        fd = ex2.getFourierDescriptor(points)
        assert np.abs(fd_ref-fd).sum() < 0.01, "There is an error with the Fourier Descriptor"
        return True
    except AssertionError as e:
        print(e)
        return False




def checkNormalizeFourierDescriptor():
    try:
        n_freq = 21
        points = np.random.random((20, 2))
        fd = points[:, 0] + 1j * points[:, 1]
        fd = np.fft.fft(fd)
        nfd = ex2.normalizeFourierDescriptor(fd, n_freq)

        assert len(nfd) == n_freq, "Fourier Descriptor has wrong size. Size should be n_freq."
        assert nfd.dtype == np.float64 or nfd.dtype == np.float32, "Fourier Descriptor is not correctly normalized. " \
                                                                   "The FD should only contain real numbers (float32 " \
                                                                   "or float64) "
        assert nfd[0] == 0, "Fourier Descriptor is not translation invariant"
        assert np.abs(nfd[1] - 1) < 0.001, "Fourier Descriptor is not scale invariant"
        return True
    except AssertionError as e:
        print(e)
        return False

def checkClassification():
    try:
        template1 = cv2.imread("data/template1.jpg")
        utils.show(template1)
        
        template2 = cv2.imread("data/template2.jpg")
        utils.show(template2)
        templates = [template1, template2]
    
    
        img = cv2.imread("data/query.jpg")
        utils.show(img)
    
        n_freq = 30
        thresh = 0.5
    
        _, labels = ex2.imageSegmentation(img, templates, n_freq, thresh)
        class1 = labels[:,:,0]==255
        class2 = labels[:,:,1]==255
        utils.show(class1)
        utils.show(class2)
    
        gt = cv2.imread("data/labels.png")
        class1_gt = gt[:,:,0]==255
        class2_gt = gt[:,:,1]==255
        utils.show(class1_gt)
        utils.show(class2_gt)
        
        intersection1 = (class1 & class1_gt & (class1_gt == class1)).sum()
        union1 = (class1 | class1_gt).sum()
        
        intersection2 = (class2 & class2_gt & (class2_gt == class2)).sum()
        union2 = (class2 | class2_gt).sum()
        
        utils.show((class1 | class1_gt))
        utils.show((class2 | class2_gt))
        
        iou1 = intersection1/union1
        iou2 = intersection2/union2
    
        assert iou1 > 0.7, "The classification of class 1 is incorrect. Score: {:2.2f}%".format(100*iou1)
        assert iou2 > 0.55, "The classification of class 2 is incorrect. Score: {:2.2f}%".format(100*iou2)
        return True
    
    except AssertionError as e:
        print(e)
        return False


if __name__ == '__main__':
    img = cv2.imread("data/query.jpg")
    ref = cv2.imread("data/binary_query.png")[:,:,0]
    
    binarization = 0
    fd = 0
    nfd = 0
    classification = 0
    
    try:
        binarization += checkBinarization(img, ref)
        fd += checkFourierDescriptor()
        nfd += checkNormalizeFourierDescriptor()
        classification += checkClassification()
    except Exception as e:
        print(e)
    
    score = { "scores": {"Binarization": binarization,
                         "FourierDescriptor": fd,
                         "NormalizeFourierDescriptor": nfd,
                         "Classification": classification} }
    
    print(json.dumps(score))

