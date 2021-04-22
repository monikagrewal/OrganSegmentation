import model.transform.transformers as transformers
from config import config

AUGMENTATION_OPTIONS = dict(
    augmentation_brightness={1: dict(p=0.5, rel_addition_range=(-0.2, 0.2))},
    augmentation_contrast={1: dict(p=0.5, contrast_mult_range=(0.8, 1.2))},
    augmentation_rotate3d={
        1: dict(p=0.3, x_range=(-20, 20), y_range=(0, 0), z_range=(0, 0)),
        2: dict(p=0.3, x_range=(-10, 10), y_range=(0, 0), z_range=(0, 0)),
        3: dict(p=0.3, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10)),
    },
)


def get_transform_pipelines() -> dict[str, transformers.Compose]:
    # Random augmentations
    transform_any = transformers.ComposeAnyOf([])
    if config.AUGMENTATION_BRIGHTNESS != 0:
        brightness_settings = AUGMENTATION_OPTIONS["augmentation_brightness"][
            config.AUGMENTATION_BRIGHTNESS
        ]
        print(
            f"Adding random brightness augmentation with params: {brightness_settings}"
        )
        transform_any.transforms.append(
            transformers.RandomBrightness(**brightness_settings)
        )
    if config.AUGMENTATION_CONTRAST != 0:
        contrast_settings = AUGMENTATION_OPTIONS["augmentation_contrast"][
            config.AUGMENTATION_CONTRAST
        ]
        print(f"Adding random contrast augmentation with params: {contrast_settings}")
        transform_any.transforms.append(
            transformers.RandomContrast(**contrast_settings)
        )
    if config.AUGMENTATION_ROTATE3D != 0:
        rotate3d_settings = AUGMENTATION_OPTIONS["augmentation_rotate3d"][
            config.AUGMENTATION_ROTATE3D
        ]
        print(f"Adding random rotate3d augmentation with params: {rotate3d_settings}")
        transform_any.transforms.append(
            transformers.RandomRotate3D(**rotate3d_settings)
        )

    # Training pipeline
    transform_train = transformers.Compose(
        [
            transform_any,
            transformers.CropDepthwise(
                crop_size=config.IMAGE_DEPTH, crop_mode="random"
            ),
        ]
    )

    # Validation pipelines
    transform_val = transformers.Compose(
        [transformers.CropDepthwise(crop_size=config.IMAGE_DEPTH, crop_mode="random")]
    )

    transform_val_sliding_window = transformers.Compose(
        [
            # transformers.CustomResize(output_size=image_size),
            # transformers.CropInplane(crop_size=crop_inplane, crop_mode='center'),
        ]
    )

    # temporary addition to test inplance scaling
    if config.IMAGE_SCALE_INPLANE is not None:
        transform_train.transforms.append(
            transformers.CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )
        transform_val.transforms.append(
            transformers.CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )
        transform_val_sliding_window.transforms.append(
            transformers.CustomResize(scale=config.IMAGE_SCALE_INPLANE)
        )

    return {
        "train": transform_train,
        "validation": transform_val,
        "validation_sliding": transform_val_sliding_window,
    }
