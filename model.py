import torch
import os
import torchvision.transforms as transforms
import numpy as np
import cv2

from Loading_Dataset import DIV2K_dataloader
from SR_NN import EDSR
from Train_and_Test import Trainer
import Image_util
from model_utils import *
    
def create_model(scale):
    """
    create an EDSR model of a specific scale

    :param scale: scale for the model
    :return: the EDSR model
    """
    return EDSR(scale=scale)

def evaluate_model(model, input_path, cuda=False):
    """
    predict the super-resolution image

    :param model: EDSR model provided
    :param input_path: the path to the input image
    :param cuda: flag indicating whether GPU will be used
    :return: input and predicted images
    """
    # convert the tensor image from 0-1 to 0-256
    input_image = 255 * transforms.ToTensor()(cv2.imread(input_path))
    if cuda:  # in case GPU will be used
        input_image = input_image.cuda()

    output_image = model(input_image)
    return input_image.detach(), output_image.detach()


def enhance(scale, image_path, pre_train=False, weight_path=None, display=False, save=False,
            output_path=None, cuda=False):
    """
    predict(display/save) the super-resolution image given the input image provided

    :param scale: model scale
    :param image_path: path to the input image
    :param pre_train: flag indicating whether a pretrained model will be used
    :param weight_path: path to the pretrained weight
    :param display: flag indicating whether the output will be displayed
    :param save: flag indicating whether the output will be saved
    :param output_path: output path of the super-resolution image
    :param cuda: flag indicating whether GPU will be used
    """
    print('initializing model ...')
    sr_model = create_model(scale=scale)  # create model of given scale

    if pre_train:  # use pretrained model
        if scale == 2:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx2.pt')
        elif scale == 3:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx3.pt')
        elif scale == 4:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx4.pt')
        else:
            raise NotImplementedError
    else:  # use self-trained checkpoint
        sr_model, _, _, _, _ = load_checkpoint(sr_model, weight_path)

    #if cuda:  # in case GPU will be used
    #    sr_model.cuda()

    # set the model to evaluation mode
    sr_model.eval()

    print('enhancing image ...')
    lr_img, sr_img = \
        evaluate_model(sr_model, image_path, cuda=cuda)

    # convert the tensor image from 0-256 to 0-1
    lr_img /= 255
    sr_img /= 255

    print('getting evaluation scores ...')
    resize_lr_img = Image_util.tensor_to_numpy(lr_img).astype(np.uint8)
    resize_lr_img = cv2.resize(resize_lr_img, dsize=(sr_img.shape[2], sr_img.shape[1]),
                              interpolation=cv2.INTER_CUBIC)
    resize_lr_img = transforms.ToTensor()(resize_lr_img)

    psnr_score = Image_util.get_psnr(resize_lr_img, sr_img, tensor=True)
    ssim_score = Image_util.get_ssim(resize_lr_img, sr_img, tensor=True)

    if display:
        Image_util.show_tensor_img(lr_img, 'low resolution image')
        Image_util.show_tensor_img(sr_img, 'super resolution image')

    if save:
        print('saving image ...')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        Image_util.save_tensor_img(sr_img,
                                   '{}_sr_x{}'.format(image_path.split('/')[-1].split('.')[0],
                                                      scale),
                                   output_path)

    print('PSNR score is: {}'.format(round(psnr_score, 2)))
    print('SSIM score is: {}'.format(round(ssim_score, 2)))
    
def train_model( train_dataloader, sr_model, epoch, lr, batch_size, checkpoint_save_path, checkpoint=False,
                checkpoint_load_path=None, cuda=False):
    """
    train the model

    :param scale: model scale
    :param train_dataloader: the dataloader of the dataset object used for training
    :param epoch: epoch number for the training
    :param lr: learning rate for the training
    :param batch_size: batch size for the dataloader
    :param checkpoint_save_path: the output path for the checkpoint
    :param checkpoint: flag indicating whether a pretrained checkpoint will be used
    :param checkpoint_load_path: the path to the checkpoint
    :param cuda: flag indicating whether GPU will be used
    """
    print('initializing model ...')
    if cuda:
        sr_model.cuda()
    # create a Trainer object
    trainer = Trainer(train_dataloader, sr_model)
    trainer.set_checkpoint_saving_path(checkpoint_save_path)
    trainer.train(epoch=epoch, checkpoint_load_path=checkpoint_load_path,
                  checkpoint=checkpoint)


if __name__ == '__main__':
    dataloader = DIV2K_dataloader
    sr_model = EDSR(scale=4)
    print('loading dataset ...')
    # make sure you change the train_root to the path of the training data directory,
    # and change the target_root to the path of the target directory

    # train the model
    train_model( train_dataloader=dataloader,sr_model=sr_model, epoch=100, lr=1e-4, batch_size=8,
                checkpoint_save_path='checkpoints/edsr4x.pt', checkpoint=False, checkpoint_load_path=None,
                cuda=False)
    
    # make sure you change the image_path to the path of the input image
    enhance(scale=4, image_path='input_image_path', pre_train=True, weight_path=None, display=True,
            save=True, output_path='results', cuda=False)