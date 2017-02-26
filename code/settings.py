main_structures_file = 'structure_dict.dat'
main_structure_name = 'radiomics_gtv'

path_to_training_set = '/home/tomasz/dev/tumor-search/training_set'
path_to_testing_set = '/home/tomasz/dev/tumor-search/testing_set'
patient_structure_file = 'structures.dat'
patient_auxiliary = 'auxiliary'
patient_contours = 'contours'
patient_pngs = 'pngs'

models_path = '/home/tomasz/dev/tumor-search/models'

ct_tags = {'scale': '(0028.0030)', 'centre': '(0020.0032)'}

image_size = 512
num_classes = 1
num_epochs = 1000
epoch_size = 1500
batch_size = 12
threshold = 0.9
