import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, glob
import json
import numpy as np
import nibabel as nib
import skimage
from skimage.io import imread, imsave
from scipy.ndimage import zoom, interpolation
from skimage.exposure import rescale_intensity


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




def augment_3d_image(image, target, augmentor={"rotate": [10, 20, 30]}):
    """
    The input image is in (ap, rl, cc) format
    So specify rotation angles in this format only
    """

    for transform, (prob, param) in augmentor.items():
        if np.random.random() < prob:
            if transform=="rotate":
                theta_x, theta_y, theta_z = param
                matrix = create_affine_matrix(theta_x, theta_y, theta_z, center=np.array([image.shape])//2)
                image = interpolation.affine_transform(image, matrix, order=1)
                target = interpolation.affine_transform(target, matrix, order=1)

            if transform=="brightness":
                image = image + param
                image[image > 1] = 1
                image[image < 0] = 0

            if transform=="contrast":
                image = image * param
                image[image > 1] = 1
                image[image < 0] = 0

    target[target > 0.5] = 1
    target[target <= 0.5] = 0

    return image, target


def normalize(image):
    image = (image - np.min(image)) / float(np.max(image) - np.min(image))
    return image



def apply_ww_wl(image, ww, wl):
    ub = wl + ww//2
    lb = wl - ww//2
    image[image > ub] = ub
    image[image < lb] = lb
    image = (image - lb) / float(ub - lb)
    return image


def preprocess(image, label, image_depth=16, augment=True):
    # augment image
    augmentor = {}  
    if augment:
        # rotate
        theta_x = np.random.randint(-10, 10)
        theta_y = np.random.randint(-10, 10)
        theta_z = np.random.randint(-45, 45)
        augmentor["rotate"] = (0.7, (theta_x, theta_y, theta_z))

        # brightness
        low, high = np.min(image), np.max(image)
        param = (np.random.randint(-0.2*100, 0.2*100) / 100) * high
        augmentor["brightness"] = (0.2, param)

        # contrast
        param = (np.random.randint(0.8*100, 1.2*100) / 100)
        augmentor["contrast"] = (0.5, param)


    image, label = augment_3d_image(image, label, augmentor)

    # bring cc along depth dimension (D, H, W in Pytorch)
    image = image.transpose(2, 0, 1)
    label = label.transpose(2, 0, 1)
    
    # add channel axis; required for neural network training
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)

    # choose only slices equal to image_depth
    start_idx = np.random.choice(list(range(0, image.shape[1] - image_depth)), 1)[0]
    indices = list(range(start_idx, start_idx+image_depth))
    image, label = image[:, indices, :, :], label[:, indices, :, :]
    return image, label


class SpleenDataset(Dataset):
    """Spleen Dataset."""

    def __init__(self, root_dir, jsonname, image_size=128, slice_thickness=5, image_depth=16, is_training=True, augment=True, transform=None):
        """
        Args:
            root_dir (string): Directory containing data.
            jsonname (string): json filename that contains data info.
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.augment = augment
        self.image_size = image_size
        self.slice_thickness = slice_thickness
        self.image_depth = image_depth
        self.transform = transform

        jsonpath = os.path.join(root_dir, jsonname)
        self.datainfo = self.read_json(jsonpath, is_training)


    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        sample = self.datainfo[idx]
        imgname, labelname = sample["image"], sample["label"]
        imgpath = os.path.join(self.root_dir, imgname)
        labelpath = os.path.join(self.root_dir, labelname)
        image, label = self.read_nifti_images(imgpath, labelpath, inplane_size=self.image_size, slice_thickness=self.slice_thickness)
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        image, label = preprocess(image, label, image_depth=self.image_depth, augment=self.augment)
        # if self.transform is not None:
            # image, label = self.transform(image, label)

        # image = np.expand_dims(image, 0)        

        return image.astype(np.float32), label.astype(np.long)      


    def estimate_class_frequency(self, nclasses=2):
        print("calculating class frequencies...")
        counts = np.zeros(nclasses, dtype=np.float32)
        for idx in range(0, len(self.datainfo)):
            _, label = self.__getitem__(idx)
            for lbl in range(nclasses):
                n = len(label[label==lbl])
                counts[lbl] += n
                

        counts = counts / float(sum(counts))
        print("done.")
        print("Class frequencies: {}".format(counts))
        return counts


    def BatchLoader(self, batchsize=1, use_cuda=False):
        indices = np.arange(0, len(self.datainfo), dtype=np.int32)
        if self.is_training:
            np.random.shuffle(indices)

        for start in range(0, len(indices)-1, batchsize):
            images, labels = list(), list()
            for idx in indices[start : start+batchsize]:
                image, label = self.__getitem__(idx)
                images.append(image)
                labels.append(label)

            images = torch.Tensor(images)
            labels = torch.Tensor(labels).long()
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            yield images, labels


    @staticmethod
    def read_json(filepath, is_training):
        f = json.load(open(filepath, "r"))
        data = f["training"]
        test_split = int(len(data) * 0.20)

        data_dict = {"training" : data[:-test_split], "test" : data[-test_split:]}
        if is_training:
            print ("Total Training data: {}".format(len(data_dict["training"])))
            return data_dict["training"]
        else:
            print ("Total test data: {}".format(len(data_dict["test"])))
            return data_dict["test"]


    @staticmethod
    def read_nifti_images(filepath, labelpath, inplane_size=256, slice_thickness=2):
        im = nib.load(filepath)
        org_slice_thickness = im.affine[2][2]
        im = im.get_fdata()
        label = nib.load(labelpath)
        label = label.get_fdata()

        # Apply WW and WL
        WW = 400
        WL = 50
        im = apply_ww_wl(im, WW, WL)
        im = normalize(im)

        # resample to given slice thickness and inplane size
        zoom_factor = [inplane_size/float(im.shape[0]), inplane_size/float(im.shape[1]), org_slice_thickness/slice_thickness]
        im = zoom(im, zoom_factor, order=1)
        label = zoom(label, zoom_factor, order=0)

        
        # image is currently in rl, pa, cc space; convert it to ap, rl, cc space (so that we can see it right)
        im = im[:, ::-1, :]
        im = im.transpose(1,0,2)
        label = label[:, ::-1, :]
        label = label.transpose(1,0,2)

        return im, label


if __name__ == '__main__':
    out_dir = "./sanity1"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    root_dir = "../Data/Task09_Spleen"
    jsonname = "dataset.json"
    is_training = True
    alpha = 0.5
    color = [1, 0, 0]
    batchsize = 1
    train_dataset = SpleenDataset(root_dir, jsonname, is_training=is_training, augment=True)
    nbatches = 0
    for images, labels in train_dataset.BatchLoader(batchsize=batchsize):
        print (images.shape, labels.shape)

        if nbatches < 5:
            images = images.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            for i in range(images.shape[2]):
                im = images[0, 0, i, :, :]
                lbl = labels[0, 0, i, :, :]

                im = np.repeat(im.flatten(), 3).reshape(im.shape[0], im.shape[1], 3)
                lbl = np.repeat(lbl.flatten(), 3).reshape(lbl.shape[0], lbl.shape[1], 3)

                im_plus_label = (1-alpha*lbl)*im + alpha*lbl*color
                out_im = np.concatenate((im, im_plus_label), axis=1)
                imsave(os.path.join(out_dir, "iter_%d_%d.jpg"%(nbatches, i)), (out_im*255).astype(np.uint8))
        else:
            break
        
        nbatches += 1

