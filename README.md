# qtim_tools
This is a repository for QTIM image processing utilties written in Python. It is a work in progress, and we would love to get your input if anything is not working as expected. Send us a message!

To get this package up and running, clone this repository from github and run from the command line:

python setup.py install

This will install a local version of qtim_tools to your workstation. Alternatively, you could remotely install the latest public version of qtim_tools using the command

pip install qtim_tools

You may need to install some other python libraries to run this code (most likely nibabel and pydicom).

# qtim_tools.qtim_features

qtim_features is meant to extract features (size, shape, texture, intensity, etc.) from medical imaging data. It currently takes .nii or .nii.gz files as input and output, although support for other filetypes will come soon.

In order to use our feature extractor, run the following code:

__import qtim_tools__

__qtim_tools.qtim_features.extract_features(folder, outfile)__

The extract_features command looks into the provided folder and attempts to match any available .nii or .nii.gz volume with a corresponding file with the suffix "-label." For example, the file "Test_Volume.nii.gz" would be matched with the file "Test_Volume-label.nii.gz", provided they are both in the given folder. Volumes without a label will be skipped. You can choose to extract features from an entire volume using the labels=False parameter. Be mindful that extracting features from large images without labels can take a very long time, particularly for GLCM and other texture features.

Volumes with multiple labels will have texture extracted from each label separately. Output data will have a suffix corresponding to the intensity value of the label provided. For example, a file named "Test_Volume.nii" with two labels at intensities 1 and 3 will have two output rows titled "Test_Volume.nii_1" and "Test_Volume.nii_3".

Full parameter list:

extract_features(

__folder__ - The input folder containing your data. Required

__outfile__ - The output file containing your data. Must be a '.csv' file. Required.

__labels__ - Set to False to extract features from an entire image. True by default.

__features__ - A list of feature types to calculate. Available features are ['GLCM','morphology','statistics']. All calculated by default.

__levels__ - Some texture features (e.g. GLCM) require that an image be quantized into a finite amount of intensity levels. 100 levels are calculated by default in these cases.

__erode__ - You may wish to erode your labels to avoid features being influenced by border regions. You can erode by voxels separately in the [X,Y,Z] axes by submitting a list of integers. Default is no erosion, [0,0,0].

__label_suffix__ - If your label files use a different identifier than '-label', you can add that identifier here. Default is '-label'.

__mask_value__ - If your background values is not 0 for your label-maps (e.g. -1), you can change that value here. Deafult is 0.

To see some sample data, check out the files at ~\qtim_tools\test_data\test_data_features. Also try running the command __qtim_tools.qtim_features.test()__ to do a test-run of extract_features() with sample data.
