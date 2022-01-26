# Unet Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,Input,concatenate,Dropout,Activation
from tensorflow.keras.models import Model
# Encoder Block
def conv_block(inputs ,n_convs = 2,filters = 64 , 
               kernel_size = 3, kernel_initializer = 'glorot_uniform',
               activation = 'relu',padding = 'same',dropout = None,n=None):
    ''' 
    inputs : 
        prev_layer
        keras Conv2d layer parameters
        
    returns :
        n convolutional layers
    '''
    x = inputs
    for i in range(n_convs):
        x = Conv2D(filters= filters,
                   kernel_size= kernel_size,
                   kernel_initializer=kernel_initializer,
                   activation= activation,
                   padding = padding,
                   name = 'conv_'+str(i)+ '_' + str(n)
                   )(x)
    # renaming the last conv2d layer to be returned
    conv = x
    # applying max pooling to the last conv2d layer
    block = MaxPooling2D(pool_size=(2,2))(conv)
    if dropout : 
        block = Dropout(0.2)(conv)
        return block,conv
    else:
        return block,conv


def encoder(inputs,stage_filters = [64,128,256,512],bottle_neck_filters = 1024):
    ''' 
    Creates the encoder block of the Unet
    inputs : 
        inputs : tf tensor -model inputs
        stage_filters : list - number of filters for every downsampling stage
        bottle_neck_filters : int- number of bottle neck filters
    returns :
        x : encode block
        dict - convs : downsampling convolution outputs layers
    '''
    convs = {}
    x = inputs
    
    # conv blocks
    for stage in stage_filters:
        x,conv = conv_block(x,filters = stage,n=stage)
        convs['block_conv_'+str(stage)] = conv
    
    # bottle neck    
    x = Conv2D(kernel_size = (1,1),filters= bottle_neck_filters,name = 'bottle_neck')(x)
    
    return x,convs


# Decoder Block
def conv_trans_block(inputs,prev_conv,filters=64,kernel_size=(3,3),strides=(2,2),kernel_initializer='glorot_uniform',n = None):
    ''' 
    Builds an upsampling block 
    inputs :
        inputs : tensor - input from previous layer
        prev_conv : tensor - skip connection from equivilant downsampling
        Conv2D parameters
        Conv2DTranspose parameters
        
    returns :
        x : tensor - output of the upsampling block
    '''
    # up sampling
    up = Conv2DTranspose(filters = filters,
                         kernel_size = kernel_size,
                         strides = strides,padding = 'same',
                         kernel_initializer=kernel_initializer,
                         activation = 'relu',
                         name = 'transpose_conv_'+str(n)
                         )(inputs)
    # concat prev conv outputs
    x = concatenate([up,prev_conv],name = 'concatenate_layer_'+str(n))
    for i in range(2):
        x = Conv2D(filters= filters,
                   kernel_size= kernel_size,
                   kernel_initializer=kernel_initializer,
                   activation= 'relu',
                   padding = 'same',
                   name = 'conv_decoder_'+str(i)+ '_' + str(n)
                   )(x)
        
    return x


def decoder(inputs ,prev_convs ,n_classes,stage_filters = [512,256,128,64]):
    ''' 
    Builds Decoder block of the Unet
    inputs : 
        inputs : tensor - the output of the bottleneck
        prev_convs : dict - dictionary of the encoder convolutions for skip connections
        n_classes : number of classes for predictions to reshape output layer
        stage_filters : list - number of filters for each upsampling block
        
    returns :
        x : tensor - output layer of the model
    '''
    
    x = inputs
    stage = 0
    for conv in reversed(prev_convs.values()):
        x = conv_trans_block(x,conv,filters=stage_filters[stage],n = stage)
        stage+=1
        
    # Reshape output
    x = Conv2D(kernel_size = (1,1),filters= n_classes,name = 'reshaping_class_conv')(x)
    # Activation layer
    x = Activation('softmax',name = 'softmax_prediction')(x)
        
    return x


# making a unet model
def build_unet(input_shape,stage_filters=[64,128,256,512],n_classes=32,weights=None):
    ''' 
    Build a Unet model
    inputs : 
        input_shape : tuple - shape of input image
        stage_filters : list - number of filters for each downsampling stage
        n_classes : Int - number of output classes
        weights : String -  path to model weights (optional)
        
    return:
        model : Unet Tensorflow model
    '''
    
    inputs = Input(shape = input_shape,name = 'input_layer')
    
    encoder_out,convs = encoder(inputs,
                                stage_filters=stage_filters
                                )
    decoder_out = decoder(encoder_out,
                          prev_convs= convs,
                          stage_filters=list(reversed(stage_filters)),
                          n_classes=n_classes
                          )
    
    model = Model(inputs = inputs,outputs = decoder_out)
    if weights : 
        model.load_weights(weights)
    return model
