from tensorflow.keras.layers import Activation,Conv2D,Dropout,BatchNormalization,UpSampling2D,add,concatenate,Input
from tensorflow.keras.models import Model


def resolution_block(input_layer,n_convs,filters,kernel_size,padding = 'same',kernel_initializer = 'glorot_uniform',activ_relu=True,
                     batch_norm = True,dropout = None,activation = 'relu'):
    '''
    Create a fully convolutional block with parameters
    inputs :
        n_convs : number of convolutional layers
        
    outputs:
        convs : 
    '''
    x = input_layer
    for i in range(n_convs):
        x = Conv2D(filters= filters,
                   kernel_size = kernel_size ,
                   padding=padding,
                   kernel_initializer=kernel_initializer )(x)
        if batch_norm :
           x = BatchNormalization()(x)
        if activ_relu :
           x = Activation('relu')(x)  
    if dropout : 
        x = Dropout(dropout)(x)
            
    
    return x

#down sampling for high resolution convolution /medium /low
def downsample_layer(input_layer,filters,kernel_size=3,strides=(2,2),batch_norm=False,activ_relu=False):
  #downsampled_layers[resolution]["downsample_1"]
  out=Conv2D(filters= filters,
            kernel_size = kernel_size ,strides=strides,
            padding='same',use_bias=False)(input_layer)
  if batch_norm :
           out = BatchNormalization()(out)
  if activ_relu :
           out = Activation('relu')(out)  
  return out



def transition_up_fuse(layer1,layer2,filters,target_size=0,batch_norm = True,activation_fun = False):
    
    size = layer1.shape[1]
    #print(size)
    ratio = target_size/size
    layer1 = UpSampling2D(size = int(ratio))(layer1)
    layer1  = Conv2D(filters=filters,kernel_size=(1,1))(layer1)
    if batch_norm :
        layer1 = BatchNormalization()(layer1)
    output=add([layer1,layer2])
    if activation_fun:
              output=Activation('relu')(output)
    #x = Activation(activation)(x)
    
    return output

def upscale(layer,filters,target_size=0,batch_norm = True,activation_fun = False):
    
    size = layer.shape[1]
    ratio = target_size/size
    layer = UpSampling2D(size = int(ratio))(layer)
    layer  = Conv2D(filters=filters,kernel_size=(1,1))(layer)
    if batch_norm :
        layer = BatchNormalization()(layer)
    if activation_fun:
          layer=Activation('relu')(layer)
    return layer

def add_layer(layers,activation_fun=True):
   '''
   this function adds two layers must be of same shape e.g(256,256,32)&(256,256,32)
   '''
   output=add(layers)
   if activation_fun:
              output=Activation('relu')(output)
   return output


def block_1(input_layer):
    out=resolution_block(input_layer,2,64,3)
    out=resolution_block(out,1,256,3,activ_relu=False)
    return out
    
    
    
def first_block(input_layer):
    x=downsample_layer(input_layer,64,batch_norm=True,activ_relu=True)
    out1=block_1(x)
    out2=resolution_block(x,1,256,3,activ_relu=False)

    out3=add_layer([out1,out2])
    out4=block_1(out3)
    out5=add_layer([out3,out4])
    out6=block_1(out5)
    out7=add_layer([out6,out5])
    out8=block_1(out7)
    out=add_layer([out8,out7])
    return out


def block2_1(input_layer):
    out=resolution_block(input_layer,1,32,3)
    out=resolution_block(out,1,32,3,activ_relu=False)  #conv>bn>act>con>bn
    return out



def block2_2(input_layer,filters=64):
    out=resolution_block(input_layer,1,filters,3)
    out=resolution_block(out,1,filters,3,activ_relu=False)
    return out



def block_h(input_layer):
    #hbtdy b high resolution
    outh1=resolution_block(input_layer,1,32,3)
    outh2=block2_1(outh1)
    outh3=add_layer([outh1,outh2])
    outh4=block2_1(outh3)
    outh5=add_layer([outh3,outh4])
    outh6=block2_1(outh5)
    outh7=add_layer([outh5,outh6])
    outh8=block2_1(outh7)
    out=add_layer([outh7,outh8])
    
    return out


def block_m_l(input_layer,filters,downsample=True):
    #hbtdy b medium resolution
    if downsample:
          input_layer=downsample_layer(input_layer,64,kernel_size=3,strides=(2,2),batch_norm=True,activ_relu=True)
    outm2=block2_2(input_layer,filters)
    outm3=add_layer([input_layer,outm2])
    outm4=block2_2(outm3,filters)
    outm5=add_layer([outm3,outm4])
    outm6=block2_2(outm5,filters)
    outm7=add_layer([outm5,outm6])
    outm8=block2_2(outm7,filters)
    out=add_layer([outm7,outm8])
    
    return out



def build_hrnet_v1(input_shape = (512,512,3),n_classes=30,weights = None):
    '''
    Builds Hrnet
    Inputs :
        input_shape : tuple - input images shape default : (512,512,3)
        weights : String -  path to model weights - optional
        
    returns :
        Keras model
    '''
    input_=Input(shape=input_shape)
    
  #building first bock
    out_1=first_block(input_)
    
  # building second block high resolution
    out_2=block_h(out_1)   #(256,256,32)
    
    # building second block medium
    out_3=block_m_l(out_1,64)    #(128,128,64)
    out=[out_2,out_3]
    
    #building third block high
    out_4=transition_up_fuse(out_3,out_2,32,target_size=256,batch_norm=True)  #(256,256,32)
    out_5=block_h(out_4)   #(256,256,32)
    
    # building third block medium
    out_6=downsample_layer(out_4,64,batch_norm=True)  #(128,128,64)
    
  # adding the output from medium and downsampled high resolution
    medium_added=[out_3,out_6]
    out_7=add_layer(medium_added,activation_fun=False)  #(128,128,64)
    out_8=block_m_l(out_7,64,downsample=False)  #(128,128,64)
    
    ##defining third block low
    out_9=downsample_layer(out_7,128,batch_norm=True,activ_relu=True)  #(64,64,128) 
    out_10=block_m_l(out_9,128,downsample=False)  #(64,64,128)
    
    # defining fourth block high resolution
    # upsampling from low resolution(64,64,128)
    out_11=upscale(out_10,32,target_size=256,batch_norm = True,activation_fun = False) #256,256,32
    
    # upscaling from medim (128,128,64)
    out_12=upscale(out_8,32,target_size=256,batch_norm = True,activation_fun = False) #256,256,32
    
    # add layer from high ,medium low
    out_13=add_layer([out_11,out_12,out_5],activation_fun=False)  #256,256,32
    out_14= block_h(out_13)     #(256,256,32)
    
    # fourth block medium resolution
    # down sampling high
    out_15=downsample_layer(out_5,64,batch_norm=True) #(128,128,64)
    #upsampling low resoltion
    out_16=upscale(out_10,64,target_size=128,batch_norm = True,activation_fun = False) #(128,128,64)
    
    # adding down sampled ,upsampled ,medium resolution
    out_17=add_layer([out_15,out_16,out_8],activation_fun=False)  #128,128,64
    
    #building convs of medium
    out_18=block_m_l(out_17,64,downsample=False)   #128,128,64
    
    # builing low resolution of fourth block
    # downsampling high resolution
    out_19=downsample_layer(out_5,128,strides=(4,4),batch_norm=True)  #64,64,128
    
    #downsampling medium
    out_20=downsample_layer(out_8,128,batch_norm=True)  #64,64,128
    out_21=add_layer([out_19,out_20,out_10],activation_fun=False) #64,64,128
    
    #building convc of low resolution
    out_4_low=block_m_l(out_21,128,downsample=False)  #64,64,128
    
    #building very low resolution of fourth block
    out_4_vlow_0=downsample_layer(out_10,256,batch_norm=True) #32,32,256
    
    #downsampling from medium resolution
    out_4_vlow_1=downsample_layer(out_8,256,strides=(4,4),batch_norm=True) #32,32,256
    
    #downsampling from high resolution
    out_4_vlow_2=downsample_layer(out_5,256,strides=(8,8),batch_norm=True) #32,32,256
    # #adding layers
    out_4_vlow=add_layer([out_4_vlow_0,out_4_vlow_1,out_4_vlow_2],activation_fun=False) 
    # # building convs of very low
    out_4_verylow_f=block_m_l(out_4_vlow,256,downsample=False) #32,32,256
    ## upscaling medium low verylow
    #upscaling very low
    output_1=upscale(out_4_verylow_f,32,target_size=256,batch_norm = True,activation_fun = False)  #vlow
    output_2=upscale(out_4_low,32,target_size=256,batch_norm = True,activation_fun = False)  #low
    output_3=upscale(out_18,32,target_size=256,batch_norm = True,activation_fun = False)   #medium
    out_f=concatenate([output_1, output_2,output_3,out_14], axis=-1)

    final_out=upscale(out_f,filters=n_classes,target_size=512,batch_norm = True,activation_fun = False)
    output_class = Activation('softmax', name='Classification')(final_out)
    model=Model(inputs=input_,outputs=output_class)
    
    if weights : 
        model.load_weights(weights)
    return  model