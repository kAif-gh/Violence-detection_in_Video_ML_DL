import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed, shuffle
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import random
import pickle
import shutil
import numpy as np
import models
from utils import *
from dataGenerator import *
#from datasetProcess import *
from tensorflow.keras.models import load_model
from keras.utils.vis_utils  import plot_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint,LearningRateScheduler
from tensorflow.python.keras import backend as K
import pandas as pd
import argparse

def train(args):
    mode = args.mode
    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM

    dataset = args.dataset # ['Data V0','Data V1','Data V2','Data']
    initial_learning_rate = 4e-04
    batch_size = args.batchSize
    vid_len = args.vidLen  # 32
    dataset_frame_size = 320 
    frame_diff_interval = 1
    input_frame_size = 224                 ###################----------------------->############################################

    lstm_type = args.lstmType # attensepconv

    crop_dark = {'Data': (0,0) }

    #---------------------------------------------------

    epochs = args.numEpochs

    #preprocess_data = args.preprocessData

    #create_new_model = ( not args.resume )

    save_path = args.savePath

    resume_path = args.resumePath

    background_suppress = args.noBackgroundSuppression

    if resume_path == "NOT_SET":
        currentModelPath =  os.path.join(save_path , str(dataset) + '_currentModel')
        #os.makedirs(currentModelPath)
    else:
        currentModelPath = resume_path

    bestValPath =  os.path.join(save_path, str(dataset) + '_best_val_acc_Model') 
    #os.makedirs(bestValPath) 

    PretrainedPath = args.PretrainedPath
    #if PretrainedPath == "NOT_SET":
        
     #   if lstm_type == "sepconv":
            ###########################
            # PretrainedPath contains path to the model which is already trained on rwf2000 dataset. It is used to initialize training on hockey or movies dataset
            # get this model from the trained_models google drive folder that I provided in readme 
            ###########################
     #       PretrainedPath = "/gdrive/Shareddrives/data PFA2/save"   # if you are using M model
      #  else:
       #     pass
        

    resume_learning_rate = 5e-05   

    cnn_trainable = True  

    one_hot = False

    loss = 'binary_crossentropy'

    #----------------------------------------------------

    # if preprocess_data:

        # if dataset == 'Data V1':
            # rootdir= '/gdrive/Shareddrives/data PFA2/Data V1' #path of the original folder
            # classes = ['Violence','NonViolence']
            # for i in classes:
            #     os.makedirs(rootdir +'/train/' + i)
            #     os.makedirs(rootdir +'/test/' + i)
            #     source = '/gdrive/Shareddrives/data PFA2/Real Life Violence Dataset/' + i
            #    allFileNames = os.listdir(source)
            #      np.random.shuffle(allFileNames)
            #     test_ratio = 0.25
            #      train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])
            #      train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
            #      test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]
            #     for name in train_FileNames:
            #         shutil.copy(name, rootdir +'/train/' + i)
            #      for name in test_FileNames:
            #          shutil.copy(name, rootdir +'/test/' + i)
            #  convert_dataset_to_npy(src='/gdrive/Shareddrives/data PFA2/{}'.format(dataset), dest='/gdrive/Shareddrives/data PFA2/{}_npy'.format(
            #     dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size)        

    train_generator = DataGenerator(directory = r'C:\\Users\\MSI\\Desktop\\pfa jupyter\\Data npy\\train',
                                    batch_size = batch_size,
                                    data_augmentation = True,
                                    shuffle = True,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = background_suppress,
                                    target_frames = vid_len,
                                    mode = "only_differences")

    test_generator = DataGenerator(directory = r'C:\\Users\\MSI\\Desktop\\pfa jupyter\\Data npy\\test',
                                    batch_size = batch_size,
                                    data_augmentation = False,
                                    shuffle = False,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = background_suppress,
                                    target_frames = vid_len,
                                    mode = "only_differences")

    #--------------------------------------------------

    print('> cnn_trainable : ',cnn_trainable)
   # if create_new_model:
    print('> creating new model...')
    model = model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, lstm_type=lstm_type)
    optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    #print('> new model created')    
    #if dataset == "Data V1" or dataset == "Data V0":
    #    print('> loading weights pretrained on rwf dataset from', PretrainedPath)
    #    model.load_weights(PretrainedPath)
    #else:
     #   print('> getting the model from...', currentModelPath)  
     #  model =  model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, lstm_type=lstm_type)
     #  optimizer = Adam(lr=resume_learning_rate, amsgrad=True)
     #  model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
     #  model.load_weights(f'{currentModelPath}')      

    print('> Summary of the model : ')
    model.summary(line_length=140)
    print('> Optimizer : ', model.optimizer.get_config())

    dot_img_file = 'model_architecture.png'
    print('> plotting the model architecture and saving at ', dot_img_file)
    plot_model(model, to_file=dot_img_file, show_shapes=True,show_layer_names=True)

    #--------------------------------------------------

    modelcheckpoint = ModelCheckpoint(
        currentModelPath , monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
        
    modelcheckpointVal = ModelCheckpoint(
        bestValPath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')

    historySavePath = os.path.join(save_path, 'results', str(dataset))
    os.makedirs(historySavePath)
    save_training_history = SaveTrainingCurves(save_path = historySavePath)

    callback_list = [
                    modelcheckpoint,
                    modelcheckpointVal,
                    save_training_history
                    ]
                    
    callback_list.append(LearningRateScheduler(lr_scheduler, verbose = 0))
                    
    #--------------------------------------------------

    model.fit(
        steps_per_epoch=len(train_generator),
        x=train_generator,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
        callbacks= callback_list
    )

    #---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--vidLen', type=int, default=64, help='Number of frames in a clip')
    parser.add_argument('--dataset', type=str, default='Data')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--resume', help='whether training should resume from the previous checkpoint',action='store_true')
    parser.add_argument('--noBackgroundSuppression', help='whether to use background suppression on frames')
    parser.add_argument('--preprocessData',type=bool,default=False )
    parser.add_argument('--mode', type=str, default='only_differences', help='model type - both, only_frames, only_differences', choices=['both', 'only_frames', 'only_differences']) 
    parser.add_argument('--lstmType', type=str, default='sepconv', help='lstm - conv, sepconv, asepconv, 3dconvblock(use 3dconvblock instead of lstm)', choices=['sepconv','asepconv', 'conv', '3dconvblock'])
    parser.add_argument('--fusionType', type=str, default='C', help='fusion type - A for add, M for multiply, C for concat', choices=['C','A','M']) 
    parser.add_argument('--savePath', type=str, default=r'C:\\Users\\MSI\\Desktop\\pfa jupyter\\save', help='folder path to save the models')
    parser.add_argument('--PretrainedPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--resumePath', type=str, default='NOT_SET', help='path to the weights for resuming from previous checkpoint')
    parser.add_argument('--resumeLearningRate', type=float, default=5e-05, help='learning rate to resume training from')
    args = parser.parse_args()
    train(args)

main()