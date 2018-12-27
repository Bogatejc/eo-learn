import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
from warnings import warn
from eolearn.core import EOTask, FeatureType

# IF not recognizing cv2 members: https://stackoverflow.com/questions/50612169/pylint-not-recognizing-cv2-members

class Thresholding():
    """
    Task to compute thresholds of the image using basic and adaptive thresholding methods.
    Depending on the image, we can also use bluring methods that sometimes improve our results.

    With adaptive thresholding we detect edges and with basic thresholding we connect field into one area - segmentation. 
    (There is a lot of room for improvment of both methods).

    The task uses methods from cv2 library.
    """

    AVAILABLE_METHODS_ADAPTIVE_THRESHOLDING = {
        'ADAPTIVE_THRESH_MEAN_C',
        'ADAPTIVE_THRESH_GAUSSIAN_C'
    }

    AVAILABLE_METHODS_SIMPLE_THRESHOLDING = {
        'THRESH_BINARY',
        'THRESH_BINARY_INV',
        'THRESH_TRUNC',
        'THRESH_TOZERO',
        'THRESH_TOZERO_INV'
    }

    AVAILABLE_TRESHOLD_TYPE = {
        'THRESH_BINARY',
        'THRESH_BINARY_INV'
    }

    def __init__(self, feature, img, simple_th_value, simple_th_maxValue, adaptive_th='ADAPTIVE_THRESH_MEAN_C', thresh_type='THRESH_BINARY', simple_th='THRESH_BINARY', blockSize=11, c=2, mask_th=10, maxValue=255, otsu=0):
        """
        :param feature: A feature that will be used and a new feature name where data will be saved. If new name is not
                        specified it will be saved with name '<feature_name>THRESH'
                        Example: (FeatureType.DATA, 'bands') or (FeatureType.DATA, 'bands', 'thresh')
        :type feature: (FeatureType, str) or (FeatureType, str, str)
        :param img: a image that will be used 
        :type img: 2D array or 3D array
        :param adaptive_th: adaptive thresholding method, ADAPTIVE_THRESH_MEAN_C=threshold value is the mean of neighbourhood area
                                                          ADAPTIVE_THRESH_GAUSSIAN_C=threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
        :type adaptive_th: str
        :param thresh_type: thresholding type used in adaptive thresholding
        :type thresh_type: str
        :param simple_th: simple thesholding method
        :type simple_th: str
        :param blockSize: it decides the size of neighbourhood area
        :type blockSize: int (must be odd)
        :param c: a contstant which is subtracted from mean or weighted mean calculated
        :type c: int 
        :param mask_th: which values do we want on our mask 
        :type mask_th: int
        :param maxValue:
        :type maxValue: int
        :param otsu: flag that tells us if we want otsu binarization or no (0->no, 1->yes)
        :type otsu: int (0 or 1)
        :param simple_th_value: threshold value for simple threshold 
        :type simple_th_value: int
        :param simple_th_maxValue: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        :type simple_th_maxValue: int
        """
        self.feature = self._parse_features(feature, default_feature_type=FeatureType.DATA, new_names=True, rename_function='{}_THRESH'.format)
        self.img = img
        self.adaptive_th = adaptive_th
        if(self.adaptive_th not in self.AVAILABLE_METHODS_ADAPTIVE_THRESHOLDING):
            raise ValueError("Adaptive thresholding method must be one of these: {}".format(self.AVAILABLE_METHODS_ADAPTIVE_THRESHOLDING))
        self.thresh_type = thresh_type
        if(self.thresh_type not in self.AVAILABLE_TRESHOLD_TYPE):
            raise ValueError("Thresholding type must be one of these:  {}".format(self.AVAILABLE_TRESHOLD_TYPE))
        self.simple_th = simple_th
        if(self.simple_th not in self.AVAILABLE_METHODS_SIMPLE_THRESHOLDING):
            raise ValueError("Simple thresholding method must be one of these: {}".format(self.AVAILABLE_METHODS_SIMPLE_THRESHOLDING))
        self.blockSize = blockSize
        if(self.blockSize % 2 != 1):
            raise ValueError("Block size must be an odd number")
        self.c = c
        self.mask_th = mask_th
        if(self.mask_th < 0 | self.mask_th > 255):
            raise ValueError("Mask threshold must be between 0 and 255")
        self.maxValue = maxValue
        if(self.maxValue > 255 | self.maxValue < 0):
            raise ValueError("maxValue must be between 0 and 255")
        self.otsu = otsu
        if(otsu != 0 | otsu != 1):
            raise ValueError("Parameter otsu must be 0 or 1")
        self.simple_th_value = simple_th_value
        if(simple_th_value > 255 | simple_th_value < 0):
            raise ValueError("simple_th_value must be between 0 and 255")
        self.simple_th_maxValue = simple_th_maxValue
        if(simple_th_maxValue > 255 | simple_th_maxValue < 0):
            raise ValueError("simple_th_maxValue must be between 0 and 255")

    def _thresholding(self):
        if(self.adaptive_th == 'ADAPTIVE_THRESH_MEAN_C'):
            if(self.thresh_type == 'THRESH_BINARY'):
                th_adaptiv = cv.adaptiveThreshold(self.img, self.maxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, self.blockSize, self.c)
            else:
                th_adaptiv = cv.adaptiveThreshold(self.img, self.maxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, self.blockSize, self.c)
        else:
            if(self.thresh_type == 'THRESH_BINARY'):
                th_adaptiv = cv.adaptiveThreshold(self.img, self.maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, self.blockSize, self.c)
            else:
                th_adaptiv = cv.adaptiveThreshold(self.img, self.maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.blockSize, self.c)
        
        mask = (th_adaptiv < self.mask_th) * 255
        self.img[mask!=0] = (255,0,0)

        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        if(self.simple_th == 'THRESH_BINARY'):
            if(self.otsu == 0):
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_BINARY)
            else:
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_BINARY + cv.THRESH_OTSU)
        elif(self.simple_th == 'THRESH_BINARY_INV'):
            if(self.otsu == 0):
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_BINARY_INV)
            else:
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        elif(self.simple_th == 'THRESH_TOZERO'):
            if(self.otsu == 0):
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TOZERO)
            else:
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TOZERO + cv.THRESH_OTSU)
        elif(self.simple_th == 'THRESH_TOZERO_INV'):
            if(self.otsu == 0):
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TOZERO_INV)
            else:
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TOZERO_INV + cv.THRESH_OTSU)
        elif(self.simple_th == 'THRESH_TRUNC'):
            if(self.otsu == 0):
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TRUNC)
            else:
                ret, result = cv.threshold(self.img, self.simple_th_value, self.simple_th_maxValue, cv.THRESH_TRUNC + cv.THRESH_OTSU)
        else:
            result = self.img

        return result
    
    def execute(self, eopatch):
        for feature_type, feature_name, new_feature_name in self.feature:
            eopatch[feature_type][new_feature_name] = self._thresholding(eopatch[feature_type][feature_name])

        return eopatch
        

class Bluring():

    AVAILABLE_BLURING_METHODS = {
        'none',
        'medianBlur',
        'GaussianBlur',
        'bilateralFilter'
    }

    def __init__(self, img, sigmaY, borderType, blur_method='none', gKsize=(5,5), sigmaX=0, mKsize=5, d=9, sigmaColor=75, sigmaSpace=75):
        """
        :param img: a image that will be used 
        :type img: 2D array or 3D array
        :param blur_method: image blurring (smoothing) methods
        :type blur method: str

        => GaussianBlur params:
        :patam gKsize: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma
        :type gKsize: Size
        :param sigmaX: Gaussian kernel standard deviation in X direction
        :type sigmaX: double
        :param sigmaY: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height
        :type sigmaY: double
        :param borderType: pixel extrapolation method
        :type borderType: int 

        => medianBlur params:
        :param mKsize: aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7...
        :type mKsize: int

        => bilateralFilter params:
        :param d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace
        :type d: int
        :param sigmaColor: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
        :type sigmaColor: double 
        :param sigmaSpace: 	Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace
        :type sigmaSpace: double
        :param borderType: border mode used to extrapolate pixels outside of the image
        :type borderType: int 
        """
        self.img = img
        self.blur_method = blur_method
        if(self.blur_method not in self.AVAILABLE_BLURING_METHODS):
            raise ValueError("Bluring method must be one of these: {}".format(self.AVAILABLE_BLURING_METHODS))

        self.gKsize = gKsize
        if(self.gKsize % 2 != 1 | self.gKsize != 0 | self.gKsize < 0):
            raise ValueError("gKsize must be odd and positive or 0")
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

        self.mKsize = mKsize
        if(self.mKsize % 2 != 1 | self.mKsize <= 1):
            raise ValueError("mKsize must be odd and greater than 1")

        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.borderType = borderType

    def _blur(self):
        if(self.blur_method == 'bilateralFilter'):
            self.img = cv.bilateralFilter(self.img, self.d, self.sigmaColor, self.sigmaSpace, self.borderType)
            return self.img
        elif(self.blur_method == 'medianBlur'):
            self.img = cv.medianBlur(self.img, self.mKsize)
            return self.img
        elif(self.blur_method == 'GaussianBlur'):
            self.img = cv.GaussianBlur(self.img, self.gKsize, self.sigmaX, self.sigmaY, self.borderType)
            return self.img
        else: 
            return self.img