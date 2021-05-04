#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simon Matern
"""
import numpy as np
import cv2
import sys
import utils
from functools import reduce
import pickle

def binarizeImage(img):
    """
    This method creates a binary image from a colored or grayscale image. 
    To improve the performance of the segmentation all structures should be 
    separated using morphological operations, e.g. leaves should not be 
    connected with each other.

    Parameters
    ----------
    img : ndarray
        An input image with shape (H,W,C) or (H,W)

    Returns
    -------
    result : ndarray
        A binary image with background pixel = 0 and foreground pixel = 255.
        The shape should be (H,W).

    """
    
    
    # converts the image to grayscale if it was
    img = img.copy()
    if len(img.shape)>2 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh=130
    #TODO:
    # ---- Binarize Image ----
    # Given a grayscale image, compute a binary representation 
    # Background pixel = 0
    # Foreground pixel = 255
    thresh, img=cv2.threshold(img, thresh, 255,cv2.THRESH_BINARY_INV  )
    #TODO (optional):
    # ---- Erosion ----
    # Use morphological erosion to remove small connections between structures
    
    #utils.show(res)
    utils.show(img)
    return img

def extractContours(img):
    """
    This method ectracts contours from a binary images. 
    The resulting contours/shapes are stored in a list. 
    Each shape is represented by its 2d points using an ndarray of size (N,2)

    Parameters
    ----------
    img : ndarray
        A binary image

    Returns
    -------
    shapes : list[ndarray]
        A list of shapes.

    """
    contours, hierarchy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    utils.visualizeContours(img, contours)
    
    shapes = []
    for cnt in contours:
        N = len(cnt)
        shape = cnt.reshape((N,2)) 
        shapes.append(shape)
    return shapes


def getFourierDescriptor(shape):
    """
    Compute the Fourier descriptor of a shape. 
    Please use numpy.fft.fft() for the fourier transform. 
    Otherwise there might be type conflicts.
    
    See: https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
    
    Parameters
    ----------
    shape : ndarray
        A shape described by an array of size (N,2). 
        Where N is the number of points.

    Returns
    -------
    fd : ndarray
        The fourier descriptor described by an array of size N.
        Note, that the output should contain complex numbers.
        
        Also note, that the 0th frequency is at fd[0]. 
        fd[1:n/2] contains all positive frequencies.
        fd[n/2:]  contains all negative frequencise.  
        See documentation for details
    """
    
    # TODO:
    # Compute fourier descriptor of shape
    #print("shape")
   
    #print(shape)
    result=shape[:,0]+shape[:,1]*1j
    #print(result)
    #print("fourier")
    
    result=np.fft.fft(result)
    #print (result)
    return result

def normalizeFourierDescriptor(fd, n_freq):
    """
    Given a Fourier descriptor and the number of frequencies, normalize
    the Fourier Descriptor and reduce the number of frequencies to n_freq.
    
    Note, that you have to remove the higher frequencies, 
    i.e. keep n_freq/2 lowest positive frequencies and n_freq/2 lowest negative frequencies.

    Parameters
    ----------
    fd : ndarray
        An unnormalized Fourier descriptor
    n_freq : int
        number of frequencies to represent the normalized FD

    Returns
    -------
    nfd : ndarray
        A normalized Fourier descriptor. 

    """
    
    fd = fd.copy()
    
    # TODO:
    # Translation Invariance F(0) := 0
    fd[0]=0
 
    # TODO:   
    # Scale Invariance F(i) := F(i)/|F(1)|
    fd=fd/fd[1]
    
    # TODO:   
    # Rotation Invariance, starting point invariance etc.
    # F := |F|
    fd=abs(fd)
    # TODO:       
    # Filter higher frequencies
    mid=int(n_freq/2)
   
    
    fd1=fd[:(mid+1)]#positive freqencies
    fd2=fd[-mid:]
    result= np.concatenate((fd1, fd2))
    return result

def classifyFourierDescriptor(nfd, nfd_templates, thresh):
    """
    Given a single normalized Fourier descriptor of shape and a list of 
    normalized Fourier descriptor of the templates, classify the the shape.
    
    Use the following metric: d(x,y) = np.linalg.norm(x-y)/N  ,where N is the 
    length of the descriptor.

    Parameters
    ----------
    nfd : ndarray
        normalized Fourier descriptor of some shape.
    nfd_templates : list[ndarrray]
        list of normalized Fourier descriptor of the templates.
    thresh : float
        a classification is only considered if the metric is below this threshold.
        Otherwise return -1

    Returns
    -------
    index : int
        The index of the corresponding class. 
        0 is the first class (first template), 1 is the second class (second template)..
        If the dissimilarity is too large return -1 
    """
    counter=-1
    last=1
    for classification in nfd_templates:
        if np.linalg.norm(nfd-classification)/len(nfd)<last:
            last= np.linalg.norm(nfd-classification)/len(nfd)
            counter=counter+1
        
        
    
    if last>thresh:
        return -1
    
    return counter


def imageSegmentation(img, templates, n_freq, thresh):
    query = binarizeImage(img)
    shapes = extractContours(query)
    shapes = [shape for shape in shapes if len(shape) > n_freq]


    utils.visualizeShapes(img, shapes, (255,255,255))
    fds = [getFourierDescriptor(shape) for shape in shapes]
    nfds = [normalizeFourierDescriptor(fd, n_freq) for fd in fds]

    nfd_templates = []
    for template in templates:
        binary = binarizeImage(template)
        templ_shapes = extractContours(binary)
        templ_shapes = [shape for shape in templ_shapes if len(shape) > n_freq]
        
        if len(templ_shapes) == 0:
            continue
        largest_shape = reduce((lambda x, y: x if len(x) > len(y) else y), templ_shapes)
        fd = getFourierDescriptor(largest_shape)
        nfd = normalizeFourierDescriptor(fd, n_freq)
        nfd_templates.append(nfd)

    classification = [classifyFourierDescriptor(nfd, nfd_templates, thresh) for nfd in nfds]

    classes = []
    labels = np.zeros(img.shape, np.uint8)
    for i in range(len(templates)):
        class_i = list(filter(lambda x: x[1] == i, enumerate(classification)))
        shapes_i = [shapes[x[0]] for x in class_i]

        color = (0,0,0)
        if i == 0:
            color = (255,0,0)
        elif i == 1:
            color = (0,255,0)
        else:
            color = (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255))
        labels = utils.visualizeShapes(labels, shapes_i, color)
        classes.append(shapes_i)

    return classes, labels

if __name__=="__main__":    
    template1 = cv2.imread("data/template1.jpg")
    utils.show(template1)
    
    template2 = cv2.imread("data/template2.jpg")
    utils.show(template2)
    templates = [template1, template2]


    img = cv2.imread("data/query.jpg")
    utils.show(img)

    n_freq = 30
    thresh = 0.5
    
    imageSegmentation(img, templates, n_freq, thresh)

    
        
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simon Matern
"""
import numpy as np
import cv2
import sys
import utils
from functools import reduce
import pickle

def binarizeImage(img):
    """
    This method creates a binary image from a colored or grayscale image. 
    To improve the performance of the segmentation all structures should be 
    separated using morphological operations, e.g. leaves should not be 
    connected with each other.

    Parameters
    ----------
    img : ndarray
        An input image with shape (H,W,C) or (H,W)

    Returns
    -------
    result : ndarray
        A binary image with background pixel = 0 and foreground pixel = 255.
        The shape should be (H,W).

    """
    
    
    # converts the image to grayscale if it was
    img = img.copy()
    if len(img.shape)>2 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh=130
    #TODO:
    # ---- Binarize Image ----
    # Given a grayscale image, compute a binary representation 
    # Background pixel = 0
    # Foreground pixel = 255
    thresh, img=cv2.threshold(img, thresh, 255,cv2.THRESH_BINARY_INV  )
    #TODO (optional):
    # ---- Erosion ----
    # Use morphological erosion to remove small connections between structures
    
    #utils.show(res)
    utils.show(img)
    return img

def extractContours(img):
    """
    This method ectracts contours from a binary images. 
    The resulting contours/shapes are stored in a list. 
    Each shape is represented by its 2d points using an ndarray of size (N,2)

    Parameters
    ----------
    img : ndarray
        A binary image

    Returns
    -------
    shapes : list[ndarray]
        A list of shapes.

    """
    contours, hierarchy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    utils.visualizeContours(img, contours)
    
    shapes = []
    for cnt in contours:
        N = len(cnt)
        shape = cnt.reshape((N,2)) 
        shapes.append(shape)
    return shapes


def getFourierDescriptor(shape):
    """
    Compute the Fourier descriptor of a shape. 
    Please use numpy.fft.fft() for the fourier transform. 
    Otherwise there might be type conflicts.
    
    See: https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
    
    Parameters
    ----------
    shape : ndarray
        A shape described by an array of size (N,2). 
        Where N is the number of points.

    Returns
    -------
    fd : ndarray
        The fourier descriptor described by an array of size N.
        Note, that the output should contain complex numbers.
        
        Also note, that the 0th frequency is at fd[0]. 
        fd[1:n/2] contains all positive frequencies.
        fd[n/2:]  contains all negative frequencise.  
        See documentation for details
    """
    
    # TODO:
    # Compute fourier descriptor of shape
    #print("shape")
   
    #print(shape)
    result=shape[:,0]+shape[:,1]*1j
    #print(result)
    #print("fourier")
    
    result=np.fft.fft(result)
    #print (result)
    return result

def normalizeFourierDescriptor(fd, n_freq):
    """
    Given a Fourier descriptor and the number of frequencies, normalize
    the Fourier Descriptor and reduce the number of frequencies to n_freq.
    
    Note, that you have to remove the higher frequencies, 
    i.e. keep n_freq/2 lowest positive frequencies and n_freq/2 lowest negative frequencies.

    Parameters
    ----------
    fd : ndarray
        An unnormalized Fourier descriptor
    n_freq : int
        number of frequencies to represent the normalized FD

    Returns
    -------
    nfd : ndarray
        A normalized Fourier descriptor. 

    """
    
    fd = fd.copy()
    
    # TODO:
    # Translation Invariance F(0) := 0
    fd[0]=0
 
    # TODO:   
    # Scale Invariance F(i) := F(i)/|F(1)|
    fd=fd/fd[1]
    
    # TODO:   
    # Rotation Invariance, starting point invariance etc.
    # F := |F|
    fd=abs(fd)
    # TODO:       
    # Filter higher frequencies
    mid=int(n_freq/2)
   
    
    fd1=fd[:(mid+1)]#positive freqencies
    fd2=fd[-mid:]
    result= np.concatenate((fd1, fd2))
    return result

def classifyFourierDescriptor(nfd, nfd_templates, thresh):
    """
    Given a single normalized Fourier descriptor of shape and a list of 
    normalized Fourier descriptor of the templates, classify the the shape.
    
    Use the following metric: d(x,y) = np.linalg.norm(x-y)/N  ,where N is the 
    length of the descriptor.

    Parameters
    ----------
    nfd : ndarray
        normalized Fourier descriptor of some shape.
    nfd_templates : list[ndarrray]
        list of normalized Fourier descriptor of the templates.
    thresh : float
        a classification is only considered if the metric is below this threshold.
        Otherwise return -1

    Returns
    -------
    index : int
        The index of the corresponding class. 
        0 is the first class (first template), 1 is the second class (second template)..
        If the dissimilarity is too large return -1 
    """
    counter=-1
    last=1
    for classification in nfd_templates:
        if np.linalg.norm(nfd-classification)/len(nfd)<last:
            last= np.linalg.norm(nfd-classification)/len(nfd)
            counter=counter+1
        
        
    
    if last>thresh:
        return -1
    
    return counter


def imageSegmentation(img, templates, n_freq, thresh):
    query = binarizeImage(img)
    shapes = extractContours(query)
    shapes = [shape for shape in shapes if len(shape) > n_freq]


    utils.visualizeShapes(img, shapes, (255,255,255))
    fds = [getFourierDescriptor(shape) for shape in shapes]
    nfds = [normalizeFourierDescriptor(fd, n_freq) for fd in fds]

    nfd_templates = []
    for template in templates:
        binary = binarizeImage(template)
        templ_shapes = extractContours(binary)
        templ_shapes = [shape for shape in templ_shapes if len(shape) > n_freq]
        
        if len(templ_shapes) == 0:
            continue
        largest_shape = reduce((lambda x, y: x if len(x) > len(y) else y), templ_shapes)
        fd = getFourierDescriptor(largest_shape)
        nfd = normalizeFourierDescriptor(fd, n_freq)
        nfd_templates.append(nfd)

    classification = [classifyFourierDescriptor(nfd, nfd_templates, thresh) for nfd in nfds]

    classes = []
    labels = np.zeros(img.shape, np.uint8)
    for i in range(len(templates)):
        class_i = list(filter(lambda x: x[1] == i, enumerate(classification)))
        shapes_i = [shapes[x[0]] for x in class_i]

        color = (0,0,0)
        if i == 0:
            color = (255,0,0)
        elif i == 1:
            color = (0,255,0)
        else:
            color = (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255))
        labels = utils.visualizeShapes(labels, shapes_i, color)
        classes.append(shapes_i)

    return classes, labels

if __name__=="__main__":    
    template1 = cv2.imread("data/template1.jpg")
    utils.show(template1)
    
    template2 = cv2.imread("data/template2.jpg")
    utils.show(template2)
    templates = [template1, template2]


    img = cv2.imread("data/query.jpg")
    utils.show(img)

    n_freq = 30
    thresh = 0.5
    
    imageSegmentation(img, templates, n_freq, thresh)

    
        
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simon Matern
"""
import numpy as np
import cv2
import sys
import utils
from functools import reduce
import pickle

def binarizeImage(img):
    """
    This method creates a binary image from a colored or grayscale image. 
    To improve the performance of the segmentation all structures should be 
    separated using morphological operations, e.g. leaves should not be 
    connected with each other.

    Parameters
    ----------
    img : ndarray
        An input image with shape (H,W,C) or (H,W)

    Returns
    -------
    result : ndarray
        A binary image with background pixel = 0 and foreground pixel = 255.
        The shape should be (H,W).

    """
    
    
    # converts the image to grayscale if it was
    img = img.copy()
    if len(img.shape)>2 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh=130
    #TODO:
    # ---- Binarize Image ----
    # Given a grayscale image, compute a binary representation 
    # Background pixel = 0
    # Foreground pixel = 255
    thresh, img=cv2.threshold(img, thresh, 255,cv2.THRESH_BINARY_INV  )
    #TODO (optional):
    # ---- Erosion ----
    # Use morphological erosion to remove small connections between structures
    
    #utils.show(res)
    utils.show(img)
    return img

def extractContours(img):
    """
    This method ectracts contours from a binary images. 
    The resulting contours/shapes are stored in a list. 
    Each shape is represented by its 2d points using an ndarray of size (N,2)

    Parameters
    ----------
    img : ndarray
        A binary image

    Returns
    -------
    shapes : list[ndarray]
        A list of shapes.

    """
    contours, hierarchy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    utils.visualizeContours(img, contours)
    
    shapes = []
    for cnt in contours:
        N = len(cnt)
        shape = cnt.reshape((N,2)) 
        shapes.append(shape)
    return shapes


def getFourierDescriptor(shape):
    """
    Compute the Fourier descriptor of a shape. 
    Please use numpy.fft.fft() for the fourier transform. 
    Otherwise there might be type conflicts.
    
    See: https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
    
    Parameters
    ----------
    shape : ndarray
        A shape described by an array of size (N,2). 
        Where N is the number of points.

    Returns
    -------
    fd : ndarray
        The fourier descriptor described by an array of size N.
        Note, that the output should contain complex numbers.
        
        Also note, that the 0th frequency is at fd[0]. 
        fd[1:n/2] contains all positive frequencies.
        fd[n/2:]  contains all negative frequencise.  
        See documentation for details
    """
    
    # TODO:
    # Compute fourier descriptor of shape
    #print("shape")
   
    #print(shape)
    result=shape[:,0]+shape[:,1]*1j
    #print(result)
    #print("fourier")
    
    result=np.fft.fft(result)
    #print (result)
    return result

def normalizeFourierDescriptor(fd, n_freq):
    """
    Given a Fourier descriptor and the number of frequencies, normalize
    the Fourier Descriptor and reduce the number of frequencies to n_freq.
    
    Note, that you have to remove the higher frequencies, 
    i.e. keep n_freq/2 lowest positive frequencies and n_freq/2 lowest negative frequencies.

    Parameters
    ----------
    fd : ndarray
        An unnormalized Fourier descriptor
    n_freq : int
        number of frequencies to represent the normalized FD

    Returns
    -------
    nfd : ndarray
        A normalized Fourier descriptor. 

    """
    
    fd = fd.copy()
    
    # TODO:
    # Translation Invariance F(0) := 0
    fd[0]=0
 
    # TODO:   
    # Scale Invariance F(i) := F(i)/|F(1)|
    fd=fd/fd[1]
    
    # TODO:   
    # Rotation Invariance, starting point invariance etc.
    # F := |F|
    fd=abs(fd)
    # TODO:       
    # Filter higher frequencies
    mid=int(n_freq/2)
   
    
    fd1=fd[:(mid+1)]#positive freqencies
    fd2=fd[-mid:]
    result= np.concatenate((fd1, fd2))
    return result

def classifyFourierDescriptor(nfd, nfd_templates, thresh):
    """
    Given a single normalized Fourier descriptor of shape and a list of 
    normalized Fourier descriptor of the templates, classify the the shape.
    
    Use the following metric: d(x,y) = np.linalg.norm(x-y)/N  ,where N is the 
    length of the descriptor.

    Parameters
    ----------
    nfd : ndarray
        normalized Fourier descriptor of some shape.
    nfd_templates : list[ndarrray]
        list of normalized Fourier descriptor of the templates.
    thresh : float
        a classification is only considered if the metric is below this threshold.
        Otherwise return -1

    Returns
    -------
    index : int
        The index of the corresponding class. 
        0 is the first class (first template), 1 is the second class (second template)..
        If the dissimilarity is too large return -1 
    """
    counter=-1
    last=1
    for classification in nfd_templates:
        if np.linalg.norm(nfd-classification)/len(nfd)<last:
            last= np.linalg.norm(nfd-classification)/len(nfd)
            counter=counter+1
        
        
    
    if last>thresh:
        return -1
    
    return counter


def imageSegmentation(img, templates, n_freq, thresh):
    query = binarizeImage(img)
    shapes = extractContours(query)
    shapes = [shape for shape in shapes if len(shape) > n_freq]


    utils.visualizeShapes(img, shapes, (255,255,255))
    fds = [getFourierDescriptor(shape) for shape in shapes]
    nfds = [normalizeFourierDescriptor(fd, n_freq) for fd in fds]

    nfd_templates = []
    for template in templates:
        binary = binarizeImage(template)
        templ_shapes = extractContours(binary)
        templ_shapes = [shape for shape in templ_shapes if len(shape) > n_freq]
        
        if len(templ_shapes) == 0:
            continue
        largest_shape = reduce((lambda x, y: x if len(x) > len(y) else y), templ_shapes)
        fd = getFourierDescriptor(largest_shape)
        nfd = normalizeFourierDescriptor(fd, n_freq)
        nfd_templates.append(nfd)

    classification = [classifyFourierDescriptor(nfd, nfd_templates, thresh) for nfd in nfds]

    classes = []
    labels = np.zeros(img.shape, np.uint8)
    for i in range(len(templates)):
        class_i = list(filter(lambda x: x[1] == i, enumerate(classification)))
        shapes_i = [shapes[x[0]] for x in class_i]

        color = (0,0,0)
        if i == 0:
            color = (255,0,0)
        elif i == 1:
            color = (0,255,0)
        else:
            color = (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255))
        labels = utils.visualizeShapes(labels, shapes_i, color)
        classes.append(shapes_i)

    return classes, labels

if __name__=="__main__":    
    template1 = cv2.imread("data/template1.jpg")
    utils.show(template1)
    
    template2 = cv2.imread("data/template2.jpg")
    utils.show(template2)
    templates = [template1, template2]


    img = cv2.imread("data/query.jpg")
    utils.show(img)

    n_freq = 30
    thresh = 0.5
    
    imageSegmentation(img, templates, n_freq, thresh)

    
        
    