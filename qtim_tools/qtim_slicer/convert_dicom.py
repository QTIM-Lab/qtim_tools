import os
import glob
import fnmatch

from optparse import OptionParser

def grab_files_recursive(input_directory, regex='*'):

    """ A convenience wrapper around os.walk. Returns a list of all files in
        a directory and all of its subdirectories.
    """

    output_list = []

    for root, subFolders, files in os.walk(input_directory):
        try:
            for file in files:
                if fnmatch.fnmatch(file, regex):
                    output_list += [os.path.join(root, file)]
        except:
            continue

    return output_list

def convert_dicom(input_folder, output_folder, plugins='DICOMScalarVolumePlugin'):
    
    dicom_files = grab_files_recursive(input_folder)

    if slicer.dicomDatabase == None:
        slicer.dicomDatabase = ctk.ctkDICOMDatabase()
        slicer.dicomDatabase.openDatabase(os.path.dirname(os.path.realpath(__file__)) + "/Slicer_Dicom_Database/ctkDICOM.sql", "SLICER")
    db = slicer.dicomDatabase

    if not plugins is list:
        plugins = [plugins]

    plugins = [slicer.modules.dicomPlugins[p]() for p in plugins]

    slicer.util.quit()

    for plugin in plugins:
        # try:
        if plugin:
            loadables = plugin.examine([dicom_files])

            if len(loadables) == 0:
                print('plugin failed to interpret this series')
            else:

                patientID = db.fileValue(loadables[0].files[0],'0010,0020')
                seriesDescription = db.fileValue(loadables[0].files[0],'0008,103e')
                seriesDescription = "".join(x for x in seriesDescription if x.isalnum())
                seriesDate = db.fileValue(loadables[0].files[0],'0008,0020')
                seriesTime = db.fileValue(loadables[0].files[0],'0008,0031')
                flipAngle = db.fileValue(loadables[0].files[0],'0018,1314')
                echoTime = db.fileValue(loadables[0].files[0],'0018,0081')
                repTime = db.fileValue(loadables[0].files[0],'0018,0080')

                output_filename =  os.path.join(output_folder, patientID + '_' + seriesDescription + '_' + seriesDate + '.nrrd')

                volume = plugin.load(loadables[0])
                if volume:
                    slicer.util.saveNode(volume,output_filename)
                    slicer.util.quit()
                    return

        else:
            continue

        # except:
            # continue

    slicer.util.quit()
    return

if __name__ == '__main__':

    # try:

        parser = OptionParser()
        parser.add_option("-i", "--inputfolder", dest="InputFolder", help="Input DICOM Folder")
        parser.add_option("-o", "--outputfolder", dest="OutputFolder", help="Output Nifti Folder")
        parser.add_option("-p", "--plugins", dest="Plugins", help="DICOM Loading Plugins")
        (options, args) = parser.parse_args()
        convert_dicom(options.InputFolder, options.OutputFolder, options.Plugins)

    # except:

        # slicer.util.quit()

# /opt/Slicer-4.5.0-1-linux-amd64/Slicer --no-main-window --disable-cli-modules --python-script /home/abeers/Github/qtim_tools/qtim_tools/qtim_slicer/convert_dicom.py -i "/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA/RIDER NEURO MRI/RIDER Neuro MRI-1023805636" -o "/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA/" -p 'MultiVolumeImporterPlugin'