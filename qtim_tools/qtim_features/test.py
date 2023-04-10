import os
import extract_features


def test():

    # This code should be run from the folder above the main "qtim_tools" folder using the command "python -m qtim_tools.qtim_features.test"

	# All niftis in this folder will be processed. The program searches for a nifti file, and then checks if there is a matching labelmap file with the suffix '-label'.
	# It currently loads from some built in data from the qtim_tools project, but you can change the filepath below to anywhere.
    test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Intensity'))
    
    # If labels is set to False, the whole image will be processed. This can take a very long time for GLCM features especially, so it is best we stick to labels.
    labels = True

    # The only available features are 'GLCM', 'morphology', and 'statistics' for now.
    features = ['GLCM','morphology', 'statistics']

    # In order for GLCM to work correctly, an image has to be reduced to a set amount of gray-levels. Using all available levels in an image will most likely produce a useless result.
    # More levels will result in more intensive computation. 
    levels = 100

    # This will save a spreadsheet of all requested feature results.
    outfile = 'test_feature_results_intensity.csv'

    # If your label is for some reason masked with a value other than zero, change this parameter.
    mask_value = 0

    # The erode parameter will take [x,y,z] pixels off in each dimension. On many volumes, it is not useful to erode in the z (axial) slice because of high slice thickness.
    # Currently, the erode parameter only applies to GLCM. It does not apply to intensity statistic features, although maybe it should.
    erode = [0,0,0]

    # If overwrite is False, then the program will try to save to the chosen filename with '_copy' appended if the chosen filename already exists.
    overwrite = True

    extract_features.generate_feature_list_batch(folder=test_folder, features=features, labels=labels, levels=levels, outfile=outfile, mask_value=mask_value, erode=erode, overwrite=overwrite)

def test_parallel():
    test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_GLCM'))
    test_folder = '/home/administrator/data/tbData/tbType/TrainingSet'
    generate_feature_list_parallel(folder=test_folder, features=['GLCM','morphology', 'statistics'], labels=True, levels=100, outfile='lung_features_results_parallel_500.csv',test=False, mask_value=0, erode=[0,0,0], overwrite=True, processes=35)
    return

def parse_command_line(argv):

    # This code should be run from the folder above the main "qtim_tools" folder using the command "python -m qtim_tools.qtim_features.test"

    # All niftis in this folder will be processed. The program searches for a nifti file, and then checks if there is a matching labelmap file with the suffix '-label'.
    # It currently loads from some built in data from the qtim_tools project, but you can change the filepath below to anywhere.
    test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Intensity'))
    
    # If labels is set to False, the whole image will be processed. This can take a very long time for GLCM features especially, so it is best we stick to labels.
    labels = True

    # The only available features are 'GLCM', 'morphology', and 'statistics' for now.
    features = ['GLCM','morphology', 'statistics']

    # In order for GLCM to work correctly, an image has to be reduced to a set amount of gray-levels. Using all available levels in an image will most likely produce a useless result.
    # More levels will result in more intensive computation. 
    levels = 100

    # This will save a spreadsheet of all requested feature results.
    outfile = 'test_feature_results_intensity.csv'

    # If your label is for some reason masked with a value other than zero, change this parameter.
    mask_value = 0

    # The erode parameter will take [x,y,z] pixels off in each dimension. On many volumes, it is not useful to erode in the z (axial) slice because of high slice thickness.
    # Currently, the erode parameter only applies to GLCM. It does not apply to intensity statistic features, although maybe it should.
    erode = [0,0,0]

    # If overwrite is False, then the program will try to save to the chosen filename with '_copy' appended if the chosen filename already exists.
    overwrite = True

    extract_features.generate_feature_list_batch(folder=test_folder, features=features, labels=labels, levels=levels, outfile=outfile, mask_value=mask_value, erode=erode, overwrite=overwrite)

def test_2():

    # This code should be run from the folder above the main "qtim_tools" folder using the command "python -m qtim_tools.qtim_features.test"

    # All niftis in this folder will be processed. The program searches for a nifti file, and then checks if there is a matching labelmap file with the suffix '-label'.
    # It currently loads from some built in data from the qtim_tools project, but you can change the filepath below to anywhere.
    test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Intensity'))
    
    # If labels is set to False, the whole image will be processed. This can take a very long time for GLCM features especially, so it is best we stick to labels.
    labels = True

    # The only available features are 'GLCM', 'morphology', and 'statistics' for now.
    features = ['GLCM','morphology', 'statistics']

    # In order for GLCM to work correctly, an image has to be reduced to a set amount of gray-levels. Using all available levels in an image will most likely produce a useless result.
    # More levels will result in more intensive computation. 
    levels = 100

    # This will save a spreadsheet of all requested feature results.
    outfile = 'test_feature_results_intensity.csv'

    # If your label is for some reason masked with a value other than zero, change this parameter.
    mask_value = 0

    # The erode parameter will take [x,y,z] pixels off in each dimension. On many volumes, it is not useful to erode in the z (axial) slice because of high slice thickness.
    # Currently, the erode parameter only applies to GLCM. It does not apply to intensity statistic features, although maybe it should.
    erode = [0,0,0]

    # If overwrite is False, then the program will try to save to the chosen filename with '_copy' appended if the chosen filename already exists.
    overwrite = True

    generate_feature_list_batch(folder=test_folder, features=features, labels=labels, levels=levels, outfile=outfile, mask_value=mask_value, erode=erode, overwrite=overwrite, mode="maximal_slice")

    print('new test now')

if __name__ == '__main__':
	test_method()

