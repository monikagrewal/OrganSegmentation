import logging
import random
from typing import Dict

import numpy as np
import skimage
import torch
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.measurements import center_of_mass

from config import config


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
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "      {0}".format(t)
        format_string += "\n)"
        return format_string


class ComposeAnyOf(object):
    """Composes several transforms together and picks one.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None):
        if len(self.transforms) == 0:
            if target is not None:
                return img, target
            else:
                return img
        # pick one of the transforms at random
        t = self.transforms[np.random.randint(0, len(self.transforms))]
        if target is not None:
            return t(img, target)
        else:
            return t(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomRotate3D(object):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, x_range=(-20, 20), y_range=(0, 0), z_range=(0, 0)):
        self.p = p
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            theta_x = rand_float_in_range(*self.x_range)
            theta_y = rand_float_in_range(*self.y_range)
            theta_z = rand_float_in_range(*self.z_range)
            matrix = self._create_affine_matrix(
                theta_x, theta_y, theta_z, center=np.array([img.shape]) // 2
            )
            img = interpolation.affine_transform(img, matrix, order=1)
            if target is not None:
                target = interpolation.affine_transform(target, matrix, order=0)
        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

    @staticmethod
    def _create_affine_matrix(theta_x, theta_y, theta_z, center=np.array([0, 0, 0], dtype=np.float64)):
        """
        Input: rotation angles in degrees
        """
        theta_x *= np.pi / 180
        theta_y *= np.pi / 180
        theta_z *= np.pi / 180

        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x), 0],
                [0, np.sin(theta_x), np.cos(theta_x), 0],
                [0, 0, 0, 1],
            ]
        )

        Ry = np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y), 0],
                [0, 1, 0, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                [0, 0, 0, 1],
            ]
        )

        Rz = np.array(
            [
                [np.cos(theta_z), -np.sin(theta_z), 0, 0],
                [np.sin(theta_z), np.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        affine_matrix = np.matmul(Rz, np.matmul(Ry, Rx))
        center = center.reshape(-1, 1)
        center_homogenous = np.array([center[0], center[1], center[2], 1.], dtype=np.float64).reshape(
            -1, 1
        )
        center_rotated = np.dot(affine_matrix, center_homogenous)

        affine_matrix[:3, 3] = center.flatten() - center_rotated.flatten()[:3]

        return affine_matrix


class RandomBrightness(object):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, rel_addition_range=(-0.2, 0.2)):
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomContrast(object):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.5, contrast_mult_range=(0.8, 1.2)):
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


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
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            shape = img.shape
            dx = (
                gaussian_filter(
                    (torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0
                )
                * self.alpha
            )
            dy = (
                gaussian_filter(
                    (torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0
                )
                * self.alpha
            )
            dz = (
                gaussian_filter(
                    (torch.rand(*shape).numpy() * 2 - 1), self.sigma, cval=0
                )
                * self.alpha
            )
            x, y, z = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
            indices = (
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
                np.reshape(z + dz, (-1, 1)),
            )
            img = map_coordinates(img, indices, order=1).reshape(shape)

            if target is not None:
                target = map_coordinates(target, indices, order=0).reshape(shape)

        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomElasticTransform3D_2(object):
    """Blabla

    Args:
        p (float):
    """

    def __init__(self, p=0.75, alpha=None, sigma=None):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, target=None):
        """
        Args:
            img (Numpy Array): image to be rotated.
            target (Numpy Array): optional target image to apply the same transform to

        Returns:
            Numpy Array: Randomly rotated image.
        """
        if random.random() <= self.p:
            shape = img.shape
            x, y, z = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )

            dx = self._random_gaussian(shape[0], x, self.sigma, self.alpha)
            dy = self._random_gaussian(shape[1], y, self.sigma, self.alpha)
            dz = self._random_gaussian(shape[2], z, self.sigma, self.alpha)
            indices = (
                np.reshape(x + dx, (-1, 1)),
                np.reshape(y + dy, (-1, 1)),
                np.reshape(z + dz, (-1, 1)),
            )
            img = map_coordinates(img, indices, order=1).reshape(shape)

            if target is not None:
                target = map_coordinates(target, indices, order=0).reshape(shape)

        if target is not None:
            return img, target
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

    @staticmethod
    def _random_gaussian(shape, grid, sigma=None, alpha=None):
        """
        Helper function for RandomElasticTransform3D_2 class
        generates random gaussian field along given axis

        """
        if sigma is None:
            if shape > 32:
                sigma = torch.randint(shape // 16, shape // 8, (1, 1)).item()
            else:
                sigma = torch.randint(shape // 8, shape // 4, (1, 1)).item()
        else:
            sigma = torch.randint(sigma // 2, sigma, (1, 1)).item()

        if alpha is None:
            alpha = torch.randint(-shape // 10, shape // 10, (1, 1)).item()
        else:
            alpha = torch.randint(-alpha, alpha, (1, 1)).item()

        if abs(alpha) < shape // 20:
            alpha = shape // 20

        center = torch.randint(shape // 4, shape - shape // 4, (1, 1)).item()

        g = alpha * np.exp(-((grid - center) ** 2 / (2.0 * sigma ** 2)))

        return g


class CropDepthwise(object):
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
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class CropInplane(object):
    """Blabla

    Args:
        p (float):

    Todo:
        Currently assumes img axes: depth * in-plane axis 0 * in-plane axis 1
        Generalize to all/multiple dimensions
    """

    def __init__(self, p=1.0, crop_mode="center", crop_size=384, crop_dim=[1, 2]):
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class CropLabel(object):
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
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class CustomResize(object):
    """Blabla

    Args:
        p (float):

    Todo:
        Currently assumes img axes: depth * in-plane axis 0 * in-plane axis 1
        Generalize to all/multiple dimensions
    """

    def __init__(self, p=1.0, scale=None, output_size=None):
        self.p = p
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

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


def get_augmentation_pipelines() -> Dict[str, Compose]:
    # Random augmentations
    transform_any = ComposeAnyOf([])
    if config.AUGMENTATION_BRIGHTNESS:
        logging.info(
            "Adding random brightness augmentation with params: "
            f"{config.AUGMENTATION_BRIGHTNESS}"
        )
        transform_any.transforms.append(
            RandomBrightness(**config.AUGMENTATION_BRIGHTNESS)
        )
    if config.AUGMENTATION_CONTRAST:
        logging.info(
            "Adding random contrast augmentation with params: "
            f"{config.AUGMENTATION_CONTRAST}"
        )
        transform_any.transforms.append(RandomContrast(**config.AUGMENTATION_CONTRAST))
    if config.AUGMENTATION_ROTATE3D:
        logging.info(
            "Adding random rotate3d augmentation with params: "
            f"{config.AUGMENTATION_ROTATE3D}"
        )
        transform_any.transforms.append(RandomRotate3D(**config.AUGMENTATION_ROTATE3D))

    # Training pipeline
    transform_train = Compose(
        [
            transform_any,
            CropDepthwise(crop_size=config.IMAGE_DEPTH, crop_mode="random"),
        ]
    )

    # Validation pipelines
    transform_val_sliding_window = Compose(
        [
            # CustomResize(output_size=image_size),
            # CropInplane(crop_size=crop_inplane, crop_mode='center'),
        ]
    )

    # temporary addition to test inplance scaling
    if config.IMAGE_SCALE_INPLANE is not None:
        transform_train.transforms.append(
            CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )
        transform_val_sliding_window.transforms.append(
            CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )

    return {
        "train": transform_train,
        "validation": transform_val_sliding_window,
    }
