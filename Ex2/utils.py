#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
def show(img):
    """
    This method takes an image in numpy format and displays it.

    Parameters
    ----------
    img : a numpy array describing an image

    Returns
    -------
    None.

    """
    
    shape = img.shape
    if len(shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    if len(shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()

def visualizeContours(img, contours):
    """
    Visualise contours

    Parameters
    ----------
    img : numpy ndarray
        the contours are drawn into this image
    contours : list of ndarrays
        each element in contours is an ndarray with shape (N,1,2) which represents a contour
    """
    vis = img.copy()
    if len(vis.shape)==2:
        vis = np.stack([vis,vis,vis],axis=2)
        
    for cnt in contours:
        vis = cv2.drawContours(vis, [cnt.astype(int)], 0, (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255)), 3)
    show(vis)

def visualizeShapes(img, shapes, color):
    vis = img.copy()
    if len(vis.shape)==2:
        vis = np.stack([vis,vis,vis],axis=2)
    
    for shape in shapes:
        N = len(shape)
        cnt = shape.reshape((N,1,2))
        vis = cv2.fillPoly(vis, [cnt.astype(int)],color)
    show(vis)
    return vis    
    
    
def visualizeFourierDiscriptor(img, fds):
    
    
    contours = []
    for fd in fds:
        fd = cv2.dft(fd, cv2.DFT_INVERSE+cv2.DFT_SCALE+cv2.DFT_ROWS)
        _,N,_ = fd.shape
        contour = fd.reshape((N,1,2))
        contours.append(contour)

    visualizeContours(img, contours)