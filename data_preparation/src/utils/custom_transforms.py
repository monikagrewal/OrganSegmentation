import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import os, glob
# import json
import numpy as np
# import nibabel as nibk
# import skimage
# from skimage.io import imread, imsave
import random
from scipy.ndimage import zoom, interpolation
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def create_affine_matrix(theta_x, theta_y, theta_z, center=np.array([0, 0, 0])):
	"""
	Input: rotation angles in degrees
	"""
	theta_x *= np.pi/180
	theta_y *= np.pi/180
	theta_z *= np.pi/180

	Rx = np.array([[1, 0, 0, 0],
				[0, np.cos(theta_x), -np.sin(theta_x), 0],
				[0, np.sin(theta_x), np.cos(theta_x), 0],
				[0, 0, 0, 1]])

	Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
				[0, 1, 0, 0],
				[-np.sin(theta_y), 0, np.cos(theta_y), 0],
				[0, 0, 0, 1]])

	Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
				[np.sin(theta_z), np.cos(theta_z), 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

	
	affine_matrix = np.matmul(Rz, np.matmul(Ry, Rx))
	center = center.reshape(-1, 1)
	center_homogenous = np.array([center[0], center[1], center[2], 1]).reshape(-1, 1)
	center_rotated = np.dot(affine_matrix, center_homogenous)

	affine_matrix[:3, 3] = center.flatten() - center_rotated.flatten()[:3]
	return affine_matrix

def elastic_transform_3d(image, alpha, sigma, sampled_indices=None):
	"""Elastic deformation of images as described in [Simard2003]_.
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		Convolutional Neural Networks applied to Visual Document Analysis", in
		Proc. of the International Conference on Document Analysis and
		Recognition, 2003.
	"""

	shape = image.shape
	if not sampled_indices:
		dx = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
		dy = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
		dz = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
		x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
		indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))

	return map_coordinates(image, indices, order=1).reshape(shape), sampled_indices



# TODO: threshold mask after all transforms?

def rand_float_in_range(min_value, max_value):
    return (np.random.rand() * (max_value - min_value)) + min_value


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None):
        for t in self.transforms:
            if target is not None:
                img, target = t(img, target)
            else:
                img = t(img)
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomRotate3D(object):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, x_range=(-1,1), y_range=(-1,1), z_range=(-45,45)):
        self.p = p
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            theta_x = rand_float_in_range(*self.x_range)
            theta_y = rand_float_in_range(*self.y_range)
            theta_z = rand_float_in_range(*self.z_range)
            matrix = create_affine_matrix(theta_x, theta_y, theta_z, center=np.array([img.shape])//2)
            img = interpolation.affine_transform(img, matrix, order=1)
            if target is not None:
                target = interpolation.affine_transform(target, matrix, order=1)
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomBrightness(object):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, rel_addition_range=(-0.2,0.2)):
        self.p = p
        self.rel_addition_range = rel_addition_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            rel_addition = rand_float_in_range(*self.rel_addition_range)
            high = np.max(img)
            addition = rel_addition * high
            img = img + addition
            img[img > 1] = 1
            img[img < 0] = 0
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomContrast(object):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, contrast_mult_range=(0.8,1.2)):
        self.p = p
        self.contrast_mult_range = contrast_mult_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            multiplier = rand_float_in_range(*self.contrast_mult_range)
            img = img * multiplier
            img[img > 1] = 1
            img[img < 0] = 0
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class RandomElasticTransform3D(object):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.75, alpha=100, sigma=5):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            shape = img.shape
            dx = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0) * self.alpha
            dy = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0) * self.alpha
            dz = gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0) * self.alpha
            x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))
            img = map_coordinates(img, indices, order=1).reshape(shape)

            if target is not None:
                target = map_coordinates(target, indices, order=1).reshape(shape)

        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CropDepthwise(object):
    """Blabla

    Args:
        p (float): 

    Todo: 
        Possibly throw an error when depth is smaller than crop_size? 
    """

    def __init__(self, p=1.0, crop_mode="random", crop_size=16):
        self.p = p
        self.crop_mode = crop_mode
        self.crop_size = crop_size

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            crop_dim = 2
            if self.crop_mode == 'random':
                start_idx = np.random.choice(list(range(0, img.shape[crop_dim] - self.crop_size + 1)), 1)[0]
                end_idx = start_idx + self.crop_size
            elif self.crop_mode == 'center':
                start_idx = int((img.shape[crop_dim] / 2) - (self.crop_size/2))
                end_idx = start_idx + self.crop_size
            elif self.crop_mode =='none':
                start_idx = 0
                end_idx = img.shape[crop_dim]
            indices = list(range(start_idx, end_idx))
            slice_tuple = tuple([slice(start_idx, end_idx) if i == crop_dim else slice(None) for i in range(len(img.shape))])
            img = img[slice_tuple]
            if target is not None:
                target = target[slice_tuple]

        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)