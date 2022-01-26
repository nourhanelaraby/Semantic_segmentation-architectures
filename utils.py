import os
import cv2
import imageio 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# setting the path to our files
import os


PATH = "city_scapes"
#lablels_path = os.path.join(PATH ,'cityscapes_dict.csv')
lablels_path = 'cityscapes_dict.csv'
labels = pd.read_csv(lablels_path, index_col =0)
id2code={i:tuple(labels.loc[cl, :]) for i,cl in enumerate(labels.index)}


# changing mask
def preprocess_mask(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 720 x 960 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id  {0:(r,g,b)}
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)  #(720,960,32)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image



def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )   #(720,960,3)
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)



def trainGenerator(train_path,image_folder,mask_folder,aug_dict_img={},aug_dict_msk={},batch_size=4,image_color_mode = "rgb",
                    mask_color_mode = "rgb",target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
   
    Inputs :
        train_path : String - Root folder path
        image_folder : String - training images folder name
        mask_folder : String - labels folder name
        aug_dict :  dict - dictionary containg augmentations for train images
        aug_dict_msk : dict - dictionary containg augmentations for labels images
        batch_size : int - batch size
        image_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training images
                            default : 'rgb'
        mask_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training labels
                            default : 'rgb'
        target_size : tuple - required image size
                        default : (512,512)
        seed : int - RNG seed to fix image-label pair generation
        
    Returns :
        Generator object                  
        
        
    '''
    image_datagen = ImageDataGenerator(**aug_dict_img)
    mask_datagen = ImageDataGenerator(**aug_dict_msk)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask_img=[preprocess_mask(mask[i]) for i in range(mask.shape[0])]
        yield(img,np.array(mask_img))
        
        
        
        
def validationGenerator(val_path,image_folder,mask_folder,aug_dict_img= {},aug_dict_msk={},batch_size=4,image_color_mode = "rgb",
                    mask_color_mode = "rgb",target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    
    Inputs :
        val_path : String - Root folder path
        image_folder : String - validation images folder name
        mask_folder : String - labels folder name
        aug_dict :  dict - dictionary containg augmentations for validation images
        aug_dict_msk : dict - dictionary containg augmentations for labels images
        batch_size : int - batch size
        image_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training images
                            default : 'rgb'
        mask_color_mode : String - color mode ('rgb' , 'gray_scale' .. etc) for training labels
                            default : 'rgb'
        target_size : tuple - required image size
                        default : (512,512)
        seed : int - RNG seed to fix image-label pair generation
        
    Returns :
        Generator object   
   
    '''
    image_datagen = ImageDataGenerator(**aug_dict_img)
    mask_datagen = ImageDataGenerator(**aug_dict_msk)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        class_mode=None,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        mask_img=[preprocess_mask(mask[i]) for i in range(mask.shape[0])]
        yield(img,np.array(mask_img))
        
        
        
def predict_visualize(image_path,model,image_size = (256,256,3),n_classes = 32,alpha = 0.7,plot = False):
    '''
    Predicts image mask and make overlayed mask for visaualization
    inputs :
        image_path : path for image to be predicted
        alpha : alpha value for mask overlay
        
    returns :
        image : ndarray - input image to the model
        pred :  ndarray - predicted RGB mask image
        pred_1h : ndarray - predicted encoding
        vis : ndarray - masked image weighted sum
        
    '''
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,image_size[:-1],interpolation = cv2.INTER_AREA)
    
    pred_1h = model.predict(np.expand_dims(image,0)/255)
    #pred_1h = np.reshape(pred_1h,(image_size[0],image_size[1],n_classes))
    pred_1h = np.squeeze(pred_1h)
    pred = onehot_to_rgb(pred_1h,id2code)
    pred_vis = np.reshape(pred,image_size)
    print(pred.shape)
    vis = cv2.addWeighted(image,1.,pred,alpha,0, dtype = cv2.CV_32F)/255
    
    if plot :    
        fig,ax = plt.subplots(1,3)
        fig.set_figwidth(20)
        fig.set_figheight(5)
        ax[0].imshow(image)
        ax[1].imshow(pred)
        ax[2].imshow(vis)
        
        ax[0].title.set_text('Image')
        ax[1].title.set_text('Predicted mask')
        ax[2].title.set_text('masked image')
    
    
    return image,pred,pred_1h,vis
        
        
        
def get_predictions_arr(images_path,model,image_size = (256,256),n_classes = 32,img_weight = 0.7,mask_weight = 0.3):
    '''
    Creates rgb and gbr predictions for sequence of images in a given folder
    inputs :
        images_path :string -  path to images folder
        model : Tensorflow model to predict
        image_size : tuple - image size default: (256,256)
        n_classes : int - number of classes
        img_weight : float - between 0-1 weight of image for the weighted sum
        mask_weight : fload - between 0-1 weight of mask for weighted sum
    return:
        bgr_list : list - contains gbr blended predictions
        rgb_list : list - contains rgb blended predictions
    
    Note : 
        image_weight + mask_weight must  = 1
    '''
    # initialize empty lists
    gbr_list = []
    rgb_list = []
    
    # loop over images in dir
    for filename in os.listdir(images_path):
        # read image
        image = cv2.imread(os.path.join(images_path,filename))
        # resized GBR image
        resized_im = cv2.resize(image,image_size,interpolation  = cv2.INTER_AREA)
        # convert to RGB for predictions
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # resized rgb image
        resized_rgb = cv2.resize(rgb,image_size,interpolation  = cv2.INTER_AREA)
        
        # Run infrence with model
        pred = model.predict(np.expand_dims(resized_rgb,0)/255)
        pred = np.reshape(pred,(image_size[0],image_size[1],n_classes))
        # convert prediction to image
        pred_vis = onehot_to_rgb(pred,id2code)
        #pred_vis = np.squeeze(pred)
        #pred_vis = np.array(pred_vis,dtype = np.uint8)
        
        # convert predicted image to BGR
        bgr = cv2.cvtColor(pred_vis,cv2.COLOR_RGB2BGR)
        # Get blended result
        vis_bgr = cv2.addWeighted(resized_im,img_weight,bgr,mask_weight,0)
        
        # append to gbr list
        gbr_list.append(vis_bgr)
        
        # convert bgr blend to RGB
        vis_rgb = cv2.cvtColor(vis_bgr,cv2.COLOR_BGR2RGB)
        # append to RGB list
        rgb_list.append(vis_rgb)
        
    return gbr_list,rgb_list
        


def createVidePrediction(model,images_folder,output_file,frame_size = (256,256),fps=7,
                          n_classes = 32,img_weight = 0.7,mask_weight = 0.3):
    '''
    Creates vedio prediction
    Inputs :
        model : Tensorflow model
        images_folder : String - Path to input images folder
        ouput_file : String - output file path/name
        frame_size : tuple -  frame size of output video , default : (256,256)
        fps : int - frames per second , default : 7
        n_classes : int - number of classes
        img_weight : float - between 0-1 weight of image for the weighted sum
        mask_weight : fload - between 0-1 weight of mask for weighted sum
        
    Note :
        image_weight + mask_weight must  = 1
    '''
    bgr,rgb = get_predictions_arr(images_path=images_folder,model=model,
                                  image_size = frame_size,n_classes = n_classes,
                                  img_weight = img_weight,mask_weight = mask_weight)
    
    out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    for im in bgr:
        out.write(im)

    out.release()



def createGifPrediction(model,images_folder,output_file,frame_size = (256,256),fps=7,
                          n_classes = 32,img_weight = 0.7,mask_weight = 0.3):
    '''
    Creates Gif prediction
    Inputs :
        model : Tensorflow model
        images_folder : String - Path to input images folder
        ouput_file : String - output file path/name
        frame_size : tuple -  frame size of output video , default : (256,256)
        fps : int - frames per second , default : 7
        n_classes : int - number of classes
        img_weight : float - between 0-1 weight of image for the weighted sum
        mask_weight : fload - between 0-1 weight of mask for weighted sum
        
    Note :
        image_weight + mask_weight must  = 1
    '''
    bgr,rgb = get_predictions_arr(images_path=images_folder,model=model,
                                  image_size = frame_size,n_classes = n_classes,
                                  img_weight = img_weight,mask_weight = mask_weight)
    
    imageio.mimsave(output_file,rgb,fps = fps)
    
    
 
