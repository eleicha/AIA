#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:01:31 2021

@author: semjon
"""

import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt

def nonMaxSuprression(img, d=5):
    """
    Given an image set all values to 0 that are not
    the maximum in this (2d+1,2d+1)-window

    Parameters
    ----------
    img : ndarray
        an image
    d : int
        for each pixels consider the surrounding (2d+1,2d+1)-window

    Returns
    -------
    result : ndarray

    """
    rows,cols = img.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            low_y = max(0, i-d)
            low_x = max(0, j-d)
            
            high_y = min(rows, i+d) 
            high_x = min(cols, j+d) 
            
            max_val = img[low_y:high_y,low_x:high_x].max()
            
            if img[i,j] == max_val:
                result[i,j] = max_val
    return result

def rotateAndScale(img, angle, scale):
    """
    Rotate and scale an image

    Parameters
    ----------
    img : ndarray
        an image
    angle : float
        angle given in degrees
    scale : float
        scaling of the image

    Returns
    -------
    result : ndarray
        a distorted image

    """
    
    h, w = img.shape
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)

    corners = np.array([[0, 0, 1],[0, h, 1], [w, 0, 1], [w, h, 1]]).T
    corners = M @ corners
    
    shift = corners.min(1)
    M[:,2]-= shift    
    
    b = corners.max(1)-corners.min(1)
    result = cv2.warpAffine(img, M, (int(b[0]),int(b[1])))
    return result

def calcDirectionalGrad(img):
    """
    Computes the gradients in x- and y-direction.
    The resulting gradients are stored as complex numbers.

    Parameters
    ----------
    img : ndarray
        an image

    Returns
    -------
    ndarray
        The array is stored in the following format: grad_x+ i*grad_y
    """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    
    return sobelx + 1.0j*sobely


def circularShift(img, dx, dy):
    """
    Performs a circular shift and puts the new origin into position (dx,dy)

    Parameters
    ----------
    img : ndarray
        an image
    dx : int
        x coordinate
    dy : int
        y coordinate

    Returns
    -------
    result : ndarray
        image with new center

    """
    img = img.copy()
    result = np.zeros_like(img)
    H,W = img.shape

    result[:-dy,:-dx] = img[dy:,dx:]
    result[:-dy,-dx:] = img[dy:,:dx]
    result[-dy:,:-dx] = img[:dy,dx:]
    result[-dy:,-dx:] = img[:dy,:dx]

    return result

def calcBinaryMask(img, thresh = 0.3):
    """
    Compute the gradient of an image and compute a binary mask
    based on the threshold. Corresponds to O^B in the slides.

    Parameters
    ----------
    img : ndarray
        an image
    thresh : float
        A threshold value. The default is 0.3.

    Returns
    -------
    binary : ndarray
        A binary image.

    """

    # TODO: 
    # -compute gradients
    grads = calcDirectionalGrad(img)

    # -threshold gradients
    betrag = np.absolute(grads)
    t = thresh*betrag.max()
    betrag[betrag<=t]=0
    betrag[betrag>t]=1

    # -return binary mask
    return betrag


def correlation(img, template):
    """
    Compute a correlation of gradients between an image and a template.
    
    Note:
    You should use the formula in the slides using the fourier transform.
    Then you are guaranteed to succeed.
    
    However, you can also compute the correlation directly. 
    The resulting image must have high positive values at positions
    with high correlation.

    Parameters
    ----------
    img : ndarray
        a grayscale image
    template : ndarray
        a grayscale image of the template

    Returns
    -------
    ndarray
        an image containing the correlation between image and template gradients.
    """
    
    # TODO:
    # -compute gradient of the image
    grads_img = calcDirectionalGrad(img)
    # -compute gradient of the template
    grads_template = calcDirectionalGrad(template)

    template_frame = np.zeros(img.shape)
    template_frame = template_frame.astype(complex)
    template_frame[:template.shape[0], :template.shape[1]] += template

    # -copy template gradient into larger frame
    frame = np.zeros(grads_img.shape)
    frame = frame.astype(complex)
    frame[:grads_template.shape[0], :grads_template.shape[1]] += grads_template
    # -apply a circular shift so the center of the original template is in the
    #   upper left corner
    shift = circularShift(frame, (int) (grads_template.shape[1]/2), (int) (grads_template.shape[0]/2))
    # -normalize template
    norm = shift/np.sum(np.absolute(shift))
    # -compute correlation //Todo: Correlation calculation how?
    #print(np.absolute(template_frame))
    #t = norm * (calcBinaryMask(template_frame))

    #utils.show(np.absolute(t))

    return np.zeros_like(img)



def GeneralizedHoughTransform(img, template, angles, scales):
    """
    Compute the generalized hough transform. Given an image and a template.
    
    Parameters
    ----------
    img : ndarray
        A query image
    template : ndarray
        a template image
    angles : list[float]
        A list of angles provided in degrees
    scales : list[float]
        A list of scaling factors

    Returns
    -------
    hough_table : list[(correlation, angle, scaling)]
        The resulting hough table is a list of tuples.
        Each tuple contains the correlation and the corresponding combination
        of angle and scaling factors of the template.
        
        Note the order of these values.
    """
    # TODO:
    # for every combination of angles and scales
    for angle, scale in zip(angles,scales):
        # -distort template
        distorted_template = rotateAndScale(template, angle, scale)
        # -compute the correlation
        cor = correlation(img, distorted_template)
        # -store results with parameters in a list


    return [(np.zeros_like(img), 0, 1)]




if __name__=="__main__":
    
    # Load query image and template 
    query = cv2.imread("data/query.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("data/template.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Visualize images
    utils.show(query)
    utils.show(template)

    # Create search space and compute GHT
    angles = np.linspace(0, 360, 36)
    scales = np.linspace(0.9, 1.3, 10)
    ght = GeneralizedHoughTransform(query, template, angles, scales)
    
    # extract votes (correlation) and parameters
    votes, thetas, s = zip(*ght)
    
    # Visualize votes
    votes = np.stack(votes).max(0)
    plt.imshow(votes)
    plt.show()

    # nonMaxSuprression
    votes = nonMaxSuprression(votes, 20)
    plt.imshow(votes)
    plt.show()

    # Visualize n best matches
    n = 10
    coords = zip(*np.unravel_index(np.argpartition(votes, -n, axis=None)[-n:], votes.shape))
    vis = np.stack(3*[query],2)
    for y,x in coords:
        print(x,y)
        vis = cv2.circle(vis,(x,y), 10, (255,0,0), 2)
    utils.show(vis)
