import logging
import random
from typing import Dict

import numpy as np
import skimage
import torch
import torch.nn.functional as F
from experiments.config import config
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import center_of_mass


def create_affine_matrix(rotation=(0, 0, 0), scale=(1, 1, 1), shear=0, translation=(0, 0, 0), center=(0, 0, 0)):
    """
    Input: rotation angles in degrees
    """
    theta_x, theta_y, theta_z = rotation
    theta_x *= np.pi/180
    theta_y *= np.pi/180
    theta_z *= np.pi/180

    Rscale = np.array([[scale[0], 0, 0, 0],
                        [0, scale[1], 0, 0],
                        [0, 0, scale[2], 0],
                        [0, 0, 0, 1]])

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

    affine_matrix = np.matmul(Rscale, np.matmul(Rz, np.matmul(Ry, Rx)))
    center = np.array(center).reshape(1, -1)
    center_homogenous = np.array([center[0, 0], center[0, 1], center[0, 2], 1]).reshape(1, -1)
    center_rotated = np.dot(center_homogenous, affine_matrix)

    translation = np.array(translation).reshape(1, -1)
    translation_homogenous = np.array([translation[0, 0], translation[0, 1], translation[0, 2], 1]).reshape(1, -1)
    translation_rotated = np.dot(translation_homogenous, affine_matrix)

    affine_matrix[3, :3] = center.flatten() - center_rotated.flatten()[:3] + translation_rotated.flatten()[:3]
    return affine_matrix


def rand_float_in_range(min_value, max_value):
    return (torch.rand((1,)).item() * (max_value - min_value)) + min_value


def random_dvf(shape, sigma=None, alpha=None):
    """
    Helper function for RandomElasticTransform3D class
    generates random dvf given axis

    """
    if sigma is None:
        sigma = rand_float_in_range(max(shape)//8, max(shape)//4)
    else:
        sigma = rand_float_in_range(sigma//2, sigma)

    if alpha is None:
        alpha = rand_float_in_range(0.01, 0.1)
    else:
        alpha = rand_float_in_range(0.01, alpha)

    g = gaussian_filter(torch.rand(*shape).numpy(), sigma, cval=0)
    g = ( (g / g.max()) * 2 - 1 ) * alpha
    return g


def random_gaussian(grid, scalar, sigma=None, alpha=None, center=None):
    """
    Helper function for RandomElasticTransform3D class
    generates random gaussian field along given axis

    """
    if sigma is None:
        sigma = rand_float_in_range(0.25*scalar, 0.5*scalar)
    else:
        sigma = rand_float_in_range(sigma//2, sigma)

    if alpha is None:
        alpha = rand_float_in_range(-0.1, 0.1)
    else:
        alpha = rand_float_in_range(-alpha, alpha)

    if abs(alpha) < 0.02:
        alpha = 0.02

    if center is None:
        center = rand_float_in_range(-0.99, 0.99) * scalar
    g = alpha * np.exp(-((grid * scalar - center)**2 / (2.0 * sigma**2)))
    return g


# TODO: threshold mask after all transforms?
def elastic_transform_3d(image, alpha, sigma, sampled_indices=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.
    """

    shape = image.shape
    if not sampled_indices:
        dx = (
            gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
        )
        dy = (
            gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
        )
        dz = (
            gaussian_filter((torch.rand(*shape).numpy() * 2 - 1), sigma, cval=0) * alpha
        )
        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        indices = (
            np.reshape(x + dx, (-1, 1)),
            np.reshape(y + dy, (-1, 1)),
            np.reshape(z + dz, (-1, 1)),
        )

    return map_coordinates(image, indices, order=1).reshape(shape), sampled_indices


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
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "      {0}".format(t)
        format_string += "\n)"
        return format_string


class ComposeAnyOf(Compose):
    """Composes several transforms together and picks one.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, target=None):
        if len(self.transforms) == 0:
            if target is not None:
                return img, target
            else:
                return img
        # pick one of the transforms at random
        t = np.random.choice(self.transforms)
        if target is not None:
            return t(img, target)
        else:
            return t(img)


class CustomTransform(object):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5):
        self.p = p
        logging.debug(
        f"Adding {self.__class__.__name__} augmentation with probability: "
        f"{self.p}")

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensorShape(CustomTransform):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img, target=None):
        if img.ndim!=3:
            raise ValueError("Expected number of dimensions for 3D = 3, got {}".format(len(img.shape)))
        else:
            img = np.expand_dims(img, axis=0)
            if target is not None:
                target = np.expand_dims(target, axis=0)
                return img, target
            else:
                return img


class AffineTransform3D(CustomTransform):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, rotation=((0,0), (0,0), (0,0)), scale=((1,1), (1,1), (1,1)), shear=(0, 0), translation=((0,0), (0,0), (0,0))):
        super().__init__(p)
        self.rotation = rotation
        self.scale = scale
        self.shear = shear
        self.translation = translation

    def __call__(self, img, target=None):
        theta_x = rand_float_in_range(*self.rotation[0])
        theta_y = rand_float_in_range(*self.rotation[1])
        theta_z = rand_float_in_range(*self.rotation[2])
        rotation = (theta_x, theta_y, theta_z)       
        
        sx = rand_float_in_range(*self.scale[0])
        sy = rand_float_in_range(*self.scale[1])
        sz = rand_float_in_range(*self.scale[2])
        scale = (sx, sy, sz)
        
        shear = rand_float_in_range(*self.shear)

        tx = rand_float_in_range(*self.translation[0])
        ty = rand_float_in_range(*self.translation[1])
        tz = rand_float_in_range(*self.translation[2])
        translation = (tx, ty, tz)

        outputs = self.affine_transform(img, target, rotation=rotation, scale=scale, shear=shear, translation=translation)
        return outputs


    def affine_transform(self, img, target=None, rotation=(0, 0, 0), scale=1, shear=0, translation=(0, 0, 0)):
        if torch.rand((1,)).item() <= self.p:
            d, h, w = img.shape
            z, y, x = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
            indices = np.array([np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1), np.ones(np.prod(img.shape))]).T  #shape N, 4

            M = create_affine_matrix(rotation=rotation, scale=scale, shear=shear, translation=translation)
            indices = np.dot(indices, M)
            # normalized grid for pytorch
            indices = indices[:, :3].reshape(d, h, w, 3)

            img = F.grid_sample(torch.tensor(img).view(1, 1, d, h, w),
                                torch.tensor(indices).view(1, d, h, w, 3), mode="bilinear", align_corners=False)
            img = img.numpy().reshape(d, h, w)

            if target is not None:
                target = F.grid_sample(torch.tensor(target).double().view(1, 1, d, h, w),
                                    torch.tensor(indices).view(1, d, h, w, 3), mode="nearest", align_corners=False)
                target = target.long().numpy().reshape(d, h, w)

        if target is not None:
            return img, target
        else:
            return img


class RandomTranslate3D(AffineTransform3D):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, translation=((-0.1,0.1), (-0.1,0.1), (-0.1,0.1))):
        super().__init__(p=p, translation=translation)


class RandomRotate3D(AffineTransform3D):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10)):
        rotation = (x_range, y_range, z_range)
        super().__init__(p=p, rotation=rotation)


class RandomScale3D(AffineTransform3D):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, scale=((0.8, 1.2), (0.8, 1.2), (0.95, 1.1))):
        super().__init__(p=p, scale=scale)


class RandomShear3D(AffineTransform3D):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.5, shear=(-20, 20)):
        super().__init__(p=p, shear=shear)


class RandomBrightness(CustomTransform):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, rel_addition_range=(-0.2, 0.2)):
        super().__init__(p)
        self.rel_addition_range = rel_addition_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            rel_addition = rand_float_in_range(*self.rel_addition_range)
            high = np.max(img)
            addition = rel_addition * high
            mask = (img >= 0.01) * (img <= high - 0.01)
            img[mask] = img[mask] + addition
            img[img > 1] = 1
            img[img < 0] = 0
        if target is not None:
            return img, target
        else:
            return img


class RandomContrast(CustomTransform):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, contrast_mult_range=(0.8, 1.2)):
        super().__init__(p)
        self.contrast_mult_range = contrast_mult_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

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


class RandomElasticTransform3DLocal(CustomTransform):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.75, alpha=None, sigma=None, mode="bilinear"):
        super().__init__(p)
        self.alpha = alpha
        self.sigma = sigma
        self.mode = mode

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if torch.rand((1,)).item() <= self.p:
            d, h, w = img.shape
            z, y, x = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')

            dz = random_gaussian(z, d/2, self.sigma, self.alpha)
            dx = random_gaussian(x, w/2, self.sigma, self.alpha)
            dy = random_gaussian(y, h/2, self.sigma, self.alpha)
            indices = np.array([np.reshape(x+dx, -1), np.reshape(y+dy, -1), np.reshape(z+dz, -1)]).T

            # normalized grid for pytorch
            indices = indices.reshape(d, h, w, 3)

            img = F.grid_sample(torch.tensor(img).view(1, 1, d, h, w),
                                torch.tensor(indices).view(1, d, h, w, 3), mode=self.mode, align_corners=False)
            img = img.numpy().reshape(d, h, w)

            if target is not None:
                target = F.grid_sample(torch.tensor(target).double().view(1, 1, d, h, w),
                                    torch.tensor(indices).view(1, d, h, w, 3), mode="nearest", align_corners=False)
                target = target.long().numpy().reshape(d, h, w)

        if target is not None:
            return img, target
        else:
            return img


class RandomElasticTransform3DOrgan(CustomTransform):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.75, mode="bilinear", organ_idx=1):
        super().__init__(p)
        self.mode = mode
        self.organ_idx = organ_idx

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        assert(target is not None)
        if torch.rand((1,)).item() <= self.p:
            organ_mask = target==self.organ_idx
            if organ_mask.any():
                d, h, w = img.shape
                z, y, x = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
                cz, cy, cx = center_of_mass(organ_mask)
                cz = cz - d/2
                cy = cy - h/2
                cx = cx - w/2

                # determine alpha and sigma based on organ size
                organ_size = (target==self.organ_idx).sum()
                sigma = (organ_size * (3/4) * (1/np.pi))**(1/3) #organ radius, assuming organ as sphere
                alpha = organ_size/target.size

                dz = random_gaussian(z, d/2, alpha=None, sigma=sigma, center=cz)
                dx = random_gaussian(x, w/2, alpha=None, sigma=sigma, center=cx)
                dy = random_gaussian(y, h/2, alpha=None, sigma=sigma, center=cy)
                indices = np.array([np.reshape(x+dx, -1), np.reshape(y+dy, -1), np.reshape(z+dz, -1)]).T

                # normalized grid for pytorch
                indices = indices.reshape(d, h, w, 3)

                img = F.grid_sample(torch.tensor(img).view(1, 1, d, h, w),
                                    torch.tensor(indices).view(1, d, h, w, 3), mode=self.mode, align_corners=False)
                img = img.numpy().reshape(d, h, w)

                if target is not None:
                    target = F.grid_sample(torch.tensor(target).double().view(1, 1, d, h, w),
                                        torch.tensor(indices).view(1, d, h, w, 3), mode="nearest", align_corners=False)
                    target = target.long().numpy().reshape(d, h, w)

        if target is not None:
            return img, target
        else:
            return img


class RandomElasticTransform3DGlobal(CustomTransform):
    """Blabla

    Args:
        p (float): 
    """

    def __init__(self, p=0.75, alpha=None, sigma=None, mode="bilinear"):
        super().__init__(p)
        self.alpha = alpha
        self.sigma = sigma
        self.mode = mode

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transformation to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if torch.rand((1,)).item() <= self.p:
            d, h, w = img.shape
            z, y, x = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')

            dx = random_dvf(img.shape, sigma=self.sigma, alpha=self.alpha)
            dy = random_dvf(img.shape, sigma=self.sigma, alpha=self.alpha)
            dz = random_dvf(img.shape, sigma=self.sigma, alpha=self.alpha)
            indices = np.array([np.reshape(x+dx, -1), np.reshape(y+dy, -1), np.reshape(z+dz, -1)]).T

            # normalized grid for pytorch
            indices = indices.reshape(d, h, w, 3)

            img = F.grid_sample(torch.tensor(img).view(1, 1, d, h, w),
                                torch.tensor(indices).view(1, d, h, w, 3), mode=self.mode, align_corners=False)
            img = img.numpy().reshape(d, h, w)

            if target is not None:
                target = F.grid_sample(torch.tensor(target).double().view(1, 1, d, h, w),
                                    torch.tensor(indices).view(1, d, h, w, 3), mode="nearest", align_corners=False)
                target = target.long().numpy().reshape(d, h, w)

        if target is not None:
            return img, target
        else:
            return img


class CropDepthwise(CustomTransform):
    """Blabla

    Args:
        p (float):

    Todo:
        Possibly throw an error when depth is smaller than crop_size?
        Generalize to all/multiple dimensions
        self.crop_mode = annotation assumes first axis of target to be depth
        handle boundary cases
    """

    def __init__(self, p=1.0, crop_mode="random", crop_size=16, crop_dim=0):
        super().__init__(p)
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.crop_dim = crop_dim

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            crop_dim = self.crop_dim
            if img.shape[crop_dim] < self.crop_size:
                pad = self.crop_size - img.shape[crop_dim]
                pad_tuple = tuple(
                    [
                        (int(np.floor(pad / 2)), int(np.ceil(pad / 2)))
                        if i == crop_dim
                        else (1, 0)
                        for i in range(len(img.shape))
                    ]
                )
                img = np.pad(img, pad_tuple, mode="constant")
                target = np.pad(target, pad_tuple, mode="constant")

            if self.crop_mode == "random":
                start_idx = np.random.choice(
                    list(range(0, img.shape[crop_dim] - self.crop_size + 1)), 1
                )[0]
                end_idx = start_idx + self.crop_size

            elif self.crop_mode == "center":
                start_idx = int((img.shape[crop_dim] / 2) - (self.crop_size / 2))
                end_idx = start_idx + self.crop_size

            elif self.crop_mode == "none":
                start_idx = 0
                end_idx = img.shape[crop_dim]

            elif self.crop_mode == "annotation":
                if target is None:
                    raise ValueError(
                        "Crop mode 'annotation' requires target to be specified"
                    )

                # choose only delineated slices
                indices = [
                    idx
                    for idx in range(img.shape[crop_dim])
                    if (target[idx, :, :] != 0).any()
                ]

                if len(indices) == 0:
                    indices = list(range(img.shape[crop_dim]))
                    # raise RuntimeError(
                    #     "No positive class in target. something is wrong."
                    # )

                center = np.random.randint(
                    indices[int(len(indices) / 2)] - 5,
                    indices[int(len(indices) / 2)] + 5,
                )
                start_idx = max(0, center - int(self.crop_size / 2))
                end_idx = start_idx + self.crop_size

                # handling corner case
                if end_idx >= img.shape[crop_dim]:
                    logging.info(
                        f"handling corner case: end_idx {end_idx} exceeds "
                        f"image dim {img.shape[crop_dim]}"
                    )
                    start_idx = img.shape[crop_dim] - self.crop_size
                    end_idx = start_idx + self.crop_size
                    logging.info(f"new values {start_idx}, {end_idx}")

            slice_tuple = tuple(
                [
                    slice(start_idx, end_idx) if i == crop_dim else slice(None)
                    for i in range(len(img.shape))
                ]
            )

            img = img[slice_tuple]
            if target is not None:
                target = target[slice_tuple]

        if target is not None:
            return img, target
        else:
            return img


class CropInplane(CustomTransform):
    """Blabla

    Args:
        p (float):

    Todo:
        Currently assumes img axes: depth * in-plane axis 0 * in-plane axis 1
        Generalize to all/multiple dimensions
    """

    def __init__(self, p=1.0, crop_mode="center", crop_size=384, crop_dim=[1, 2]):
        super().__init__(p)
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.crop_dim = crop_dim

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be transformed.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: transformed image (and optionally target).
        """
        if random.random() <= self.p:
            crop_dim = self.crop_dim
            if self.crop_mode == "random":
                start_idx = np.random.choice(
                    list(range(0, img.shape[crop_dim[0]] - self.crop_size + 1)), 1
                )[0]
                end_idx = start_idx + self.crop_size

            elif self.crop_mode == "center":
                start_idx = int((img.shape[crop_dim[0]] / 2) - (self.crop_size / 2))
                end_idx = start_idx + self.crop_size

            slice_tuple = tuple(
                [
                    slice(start_idx, end_idx) if i in crop_dim else slice(None)
                    for i in range(len(img.shape))
                ]
            )
            img = img[slice_tuple]
            if target is not None:
                target = target[slice_tuple]

        if target is not None:
            return img, target
        else:
            return img


class CropLabel(CustomTransform):
    """

    Args:
        p (float):

    Todo:
        Possibly throw an error when depth is smaller than crop_size?
        Generalize to all/multiple dimensions
        self.crop_mode = annotation assumes first axis of target to be depth
        handle boundary cases
    """

    def __init__(
        self,
        p=1.0,
        crop_sizes=(16, 256, 256),
        class_weights=[3, 1, 1, 1, 1],
        rand_transl_range=(5, 25, 25),
        bg_class_idx=0,
    ):
        super().__init__(p)
        self.crop_sizes = crop_sizes
        self.class_probs = np.array(class_weights)
        self.class_probs = self.class_probs / self.class_probs.sum()
        self.bg_class_idx = bg_class_idx
        self.rand_transl_range = rand_transl_range

    def __call__(self, img, target):
        if random.random() <= self.p:

            crop_depth, crop_height, crop_width = self.crop_sizes
            if img.shape[0] < crop_depth:
                pad = crop_depth - img.shape[0]
                pad_tuple = (
                    (int(np.floor(pad / 2)), int(np.ceil(pad / 2))),
                    (0, 0),
                    (0, 0),
                )
                img = np.pad(img, pad_tuple, mode="constant")
                if target is not None:
                    target = np.pad(target, pad_tuple, mode="constant")

            # pick random class
            random_class = np.random.choice(
                range(len(self.class_probs)), p=self.class_probs
            )

            class_mask = target == random_class
            # center of mass crop if not background, and actual class pixels are present
            if random_class != self.bg_class_idx and class_mask.sum() > 0:
                # Center of mass cropping
                cmass_tuple = center_of_mass(class_mask)
                transl_factor = (np.random.rand(len(self.rand_transl_range)) * 2) - 1
                rand_translations = transl_factor * np.array(self.rand_transl_range)
                cmass_tuple = tuple(
                    [
                        cmass + trans
                        for cmass, trans in zip(cmass_tuple, rand_translations)
                    ]
                )
                start_indici = [
                    int(cmass - crop_size / 2)
                    for cmass, crop_size in zip(cmass_tuple, self.crop_sizes)
                ]
                for i in range(len(start_indici)):
                    if start_indici[i] + self.crop_sizes[i] > target.shape[i]:
                        start_indici[i] = target.shape[i] - self.crop_sizes[i]
                    elif start_indici[i] < 0:
                        start_indici[i] = 0
                crop_indici = [
                    (start, start + crop_size)
                    for (start, crop_size) in zip(start_indici, self.crop_sizes)
                ]
            else:
                # Random cropping
                start_indici = [
                    np.random.randint(0, img.shape[i] - self.crop_sizes[i] + 1)
                    for i in range(len(self.crop_sizes))
                ]
                crop_indici = [
                    (start, start + crop_size)
                    for (start, crop_size) in zip(start_indici, self.crop_sizes)
                ]
            slice_tuple = tuple([slice(start, end) for start, end in crop_indici])
            img = img[slice_tuple]
            target = target[slice_tuple]
            return img, target


class CustomResize(CustomTransform):
    """Blabla

    Args:
        p (float):

    Todo:
        Currently assumes img axes: depth * in-plane axis 0 * in-plane axis 1
        Generalize to all/multiple dimensions
    """

    def __init__(self, p=1.0, scale=None, output_size=None):
        super().__init__(p)
        if scale is None and output_size is None:
            raise ValueError("Either scale or output_size needs to be set")
        if scale is not None and output_size is not None:
            raise ValueError("Either scale or output_size needs to be set. Not both!")
        self.output_size = output_size
        self.scale = scale

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be transformed.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: transformed image (and optionally target).
        """
        if random.random() <= self.p:
            nslices = img.shape[0]
            output_size = self.output_size
            if output_size is None:
                output_size = int(img.shape[1] * self.scale)

            new_im = np.zeros((nslices, output_size, output_size))
            for i in range(nslices):
                new_im[i, :, :] = skimage.transform.resize(
                    img[i, :, :], (output_size, output_size), mode="constant"
                )

            if target is not None:
                new_target = np.zeros((nslices, output_size, output_size))
                for i in range(nslices):
                    new_target[i, :, :] = skimage.transform.resize(
                        target[i, :, :],
                        (output_size, output_size),
                        mode="constant",
                        order=0,
                        preserve_range=True,
                    )

            if target is not None:
                return new_im, new_target
            else:
                return new_im

        else:
            if target is not None:
                return img, target
            else:
                return img


class HorizontalFlip3D(CustomTransform):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        assert(img.ndim==3)
        if random.random() <= self.p:
            img = img[:, :, ::-1]
            if target is not None:
                target = target[:, :, ::-1]
        
        if target is not None:
            return img, target
        else:
            return img


class RandomMaskOrgan(CustomTransform):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, organ_idx=1):
        super().__init__(p)
        self.organ_idx = organ_idx

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        assert(img.ndim==3)
        assert(target is not None)
        if random.random() <= self.p:
            if (target==self.organ_idx).any():
                img[target==self.organ_idx] = rand_float_in_range(img.min(), img.max())
            else:
                logging.warning(f"organ index {self.organ_idx} not present in target")
        
        if target is not None:
            return img, target
        else:
            return img
