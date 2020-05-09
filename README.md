# SDNetTransferLearning

Spatial decomposition for removing domain acquisition variant features in MR data.
For training the spatial decomposition network on your data, 

'''
python train_sdnet --config path_to_config_file
'''

To standardize images from separate modalities, 

'''
python convert_modality --config path_to_config_file
'''

Finally, to train the CNN,

'''
python train_cnn --config path_to_config_file
'''
