from __future__ import division
import numpy as np
import nibabel as nib
import dicom
import os
from shutil import copy, move
import matplotlib.pyplot as plt
import glob
from scipy import stats, signal, misc
from scipy.ndimage.morphology import binary_fill_holes
import csv
import fnmatch

def copy_files(infolder, outfolder, name, duplicate=True):

    """ I'm not sure how many of these file-moving helper functions should be
        included. I know I like to have them, but it may be better to have 
        people code their own. It's hard to customize such functions exactly
        to users' needs.
    """

    path = os.path.join(infolder, name)
    print path
    files = glob.glob(path)
    if files == []:
        print 'No files moved. Might you have made an error with the filenames?'
    else:
        for file in files:
            if duplicate:
                copy(file, outfolder)
            else:
                move(file, outfolder)

def return_dicom_dictionary(filepath=[], folder=[], attributes_regex="*.dcm"):

    """ This is currently broken, and overly-specific. Consider removing.
    """

    image_numpy = nib.load('C:/Users/azb22/Documents/Scripting/Breast_MRI_Challenge/ISPY_data/AllFiles/ISPY1_1098_19860919/ISPY1_1098_19860919_BreastTissue.nii.gz')
    transform_matrix = image_numpy.affine

    if folder != []:
        img_dicom_list = []
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, attributes_regex):
                img_dicom_list.append(os.path.join(root, filename))

        if img_dicom_list == []:
            print "No DICOM attributes returned. Folder is empty."
        else:

            attribute_list = np.zeros((1, 10), dtype=object)
            temp_list = np.zeros((1, 10), dtype=object)
            attribute_list[0,:] = ['filename','PatientName','date','center1', 'center2', 'center3', 'lx','ly','lz','type']
            RadiusMatrix = np.zeros((3,3), dtype=float)

            for filepath_idx, filepath in enumerate(img_dicom_list):
                img_dicom = dicom.read_file(filepath)
                try:
                    # print img_dicom[0x117,0x1020][0][0x117,0x1043].value
                    RadiusMatrix[0,:] = img_dicom[0x117,0x1020][0][0x117,0x1043].value[0:3]
                    RadiusMatrix[1,:] = img_dicom[0x117,0x1020][0][0x117,0x1044].value[0:3]
                    RadiusMatrix[2,:] = img_dicom[0x117,0x1020][0][0x117,0x1045].value[0:3]
                    # print filepath
                    temp_list[0, 3] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[0]
                    temp_list[0, 4] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[1]
                    temp_list[0, 5] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[2]
                    temp_list[0, 6] = np.max(abs(RadiusMatrix[:,0]))
                    temp_list[0, 7] = np.max(abs(RadiusMatrix[:,1]))
                    temp_list[0, 8] = np.max(abs(RadiusMatrix[:,2]))
                    # print filepath
                    temp_list[0, 9] = img_dicom[0x117,0x1020][0][0x117,0x1046].value
                    temp_list[0, 2] = img_dicom.StudyDate
                    temp_list[0, 1] = img_dicom.PatientID
                    temp_list[0, 0] = filepath
                    if [img_dicom.StudyDate, img_dicom.PatientID] not in attribute_list[:,1:3]:
                        attribute_list = np.vstack((attribute_list, temp_list))
                        # print temp_list
                    # print ''
                except:
                    continue
                if attribute_list[0, 7] == 'nan':
                    # continue
                    for i in xrange(attribute_list.shape[1]):
                        print attribute_list[0, i]
                    for i in xrange(2,6):
                        col_vec = attribute_list[0, i][0:3]
                        # col_vec[0]  = -col_vec[0]
                        # col_vec[1] = -col_vec[1]
                        # print np.round(nib.affines.apply_affine(transform_matrix, col_vec))
                    # print col_vec.shape
                    # for i in xrange(1,5):
                        # transformed = nib.affines.apply_affine(np.linalg.inv(transform_matrix), np.reshape(attribute_list[0, i][0:3], (1,3)))
                        # print transformed
                        # image = image_numpy.get_data()

                # fig = plt.figure()
                # imgplot = plt.imshow(image[:,:,int(transformed[0][2])], interpolation='none', aspect='auto')
                # plt.show()
                
                # print transform_matrix
                # print attribute_list[0, 1]
            with open('C:/Users/azb22/Documents/GitHub/QTIM_Pipelines/QTIM_Feature_Extraction_Pipeline/Test_Data/VOI_INFO_ISPY1.csv', 'wb') as writefile:
                csvfile = csv.writer(writefile, delimiter=',')
                for row in attribute_list:
                    csvfile.writerow(row)

        # print attribute_list

    elif filepath != []:
        img_dicom = dicom.read_file(filepath) 
        print img_dicom

    else:
        print "No DICOM attributes returned. Please provide a file or folder path."
        return

def return_nifti_attributes(filepath):

    """ For now, this just returns pixel dimensions, which are important
        for calculating volume etc. It is still unknown whether pixel
        dimensions are transformed by an affine matrix; they may need to
        be transformed back in the case of our arrays. TO-DO: Make this
        return a dictionary.
    """

    img_nifti = nib.load(filepath)
    return img_nifti.header

def nifti_2_numpy(filepath):

    """ There are a lot of repetitive conversions in the current iteration
        of this program. Another option would be to always pass the nibabel
        numpy class, which contains image and attributes. But not everyone
        knows how to use that class, so it may be more difficult to troubleshoot.
    """

    img = nib.load(filepath).get_data().astype(float)
    return img

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    image_affine = nifti_image.affine
    output_nifti = nib.Nifti1Image(image_numpy, image_affine)
    nib.save(output_nifti, output_path)

def dcm_2_numpy(filepath):

    # Maybe for the future...

    return

def get_intensity_range(image_numpy, percentiles=[.25,.75]):
    intensity_range = [np.percentile(image_numpy, .25, interpolation="nearest"), np.percentile(image_numpy, .75, interpolation="nearest")]

def histogram_normalization(image_numpy, mode='uniform'):

    return

def match_array_orientation(image1, image2):

    """ Flipping images and labels is often necessary, but also often
        idiosyncratic to the two images being flipped. It's an open question
        whether the flipping can be automatically determined. If it can, this
        function will do it; if not, other parameters will have to be added by the user.
        REMINDER: This will also have to change an image's origin, if that
        image has any hope of being padded correctly in a subsequent step.
    """

    image1_nifti = nib.load(image1)
    image2_nifti = nib.load(image2)

    image1_array = image1_nifti.get_data()
    image2_array = image2_nifti.get_data()

    image1_orientation = [image1_nifti.affine[0,0], image1_nifti.affine[1,1], image1_nifti.affine[2,2]]
    image2_orientation = [image2_nifti.affine[0,0], image2_nifti.affine[1,1], image2_nifti.affine[2,2]]

    return

def pad_nifti_image(image):

    """ Many people store their label maps in Niftis with dimensions smaller
        than the corresponding image. This is also that natural output of DICOM-SEG
        nifti conversions. Padding these arrays with empty values so that they are
        comparable requires knowledge of that image's origin.
    """

    return

def mask_nifti(image_numpy, label_numpy, label_indices, mask_value=0):

    masked_images = []

    for idx in label_indices[1:]:
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = mask_value
        masked_image = truncate_image(masked_image, mask_value)
        masked_images += [masked_image]

    return masked_images

def truncate_image(image_numpy, mask_value=0):

    """ Filed To: This Is So Stupid
        There are better ways online to do what I am attempting,
        but so far I have not gotten any of them to work. In the meantime,
        this long and probably ineffecient code will suffice. It is
        meant to remove empty rows from images.
    """

    dims = image_numpy.shape
    truncate_range_x = [0,dims[0]]
    truncate_range_y = [0,dims[1]]
    truncate_range_z = [0,dims[2]]
    start_flag = True

    for x in xrange(dims[0]):
        if (image_numpy[x,:,:] == mask_value).all():
            if start_flag:
                truncate_range_x[0] = x + 1
        else:
            start_flag = False
            truncate_range_x[1] = x + 1

    start_flag = True

    for y in xrange(dims[1]):
        if (image_numpy[:,y,:] == mask_value).all():
            if start_flag:
                truncate_range_y[0] = y + 1
        else:
            start_flag = False
            truncate_range_y[1] = y + 1

    start_flag = True

    for z in xrange(dims[2]):
        if (image_numpy[:,:,z] == mask_value).all():
            if start_flag:
                truncate_range_z[0] = z + 1
        else:
            start_flag = False
            truncate_range_z[1] = z + 1

    truncate_image_numpy = image_numpy[truncate_range_x[0]:truncate_range_x[1], truncate_range_y[0]:truncate_range_y[1], truncate_range_z[0]:truncate_range_z[1]]


    return truncate_image_numpy

def assert_3D(image_numpy):
    if len(image_numpy.shape) > 3:
        if len(image_numpy.shape) == 4 and image_numpy.shape[3] == 1:
            image_numpy = image_numpy[:,:,:,0]
        else:
            return False

    return True

def coerce_levels(image_numpy, levels=255, method="divide", reference_image = [], reference_norm_range = [.075, 1], mask_value=0, coerce_positive=True):

    """ In volumes with huge outliers, the divide method will
        likely result in many zero values. This happens in practice
        quite often. TO-DO: find a better method to bin image values.
        I'm sure there are a thousand such algorithms out there to do
        so. Maybe something based on median's, rather than means. This,
        of course, loses the 'Extremeness' of extreme values. An open
        question of how to reconcile this -- maybe best left to the user.
        Note that there is some dubious +1s and -1s in this function. It
        may be better to clean these up in the future. I have also built-in
        the coerce-positive function into this function. The other one
        was not working for mysterious reasons.
    """

    if np.min(image_numpy) < 0 and coerce_positive:
        reference_image -= np.min(image_numpy)
        image_numpy[image_numpy != mask_value] -= np.min(image_numpy)

    levels -= 1
    if method == "divide":
        if reference_image == []:
            image_max = np.max(image_numpy)
        else:
            image_max = np.max(reference_image)
        for x in xrange(image_numpy.shape[0]):
            for y in xrange(image_numpy.shape[1]):
                for z in xrange(image_numpy.shape[2]):
                    if image_numpy[x,y,z] != mask_value:
                        image_numpy[x,y,z] = np.round((image_numpy[x,y,z] / image_max) * levels) + 1

    """ Another method is to bin values based on their z-score. I provide
        two options: within-ROI normalization, and whole-image normalization.
        The output is always the ROI image, but in the latter option z-scores
        are generated from the range of intensities across the entire image
        within some range of percentages. This range is currently determined
        from the mean, but it may make more sense to do it from the median;
        this protects the algorithm from extreme values. On the other hand,
        using the median could white out an otherwise heterogenous hotspot.
    """

    if method == "z_score":

        # check_image(image_numpy, mode="maximal_slice", mask_value=mask_value)

        ## Note that this is a bad way to check this variable.
        if reference_image == []:
            masked_image_numpy = np.ma.masked_equal(image_numpy, mask_value)
            z_image_numpy = stats.zscore(masked_image_numpy, axis=None)

            # image_range = [np.min(z_image_numpy), np.max(z_image_numpy)]
            image_range = [np.mean(z_image_numpy) - np.std(z_image_numpy), np.mean(z_image_numpy) + np.std(z_image_numpy)]
            bins = np.linspace(image_range[0], image_range[1], levels)

            # distribution = stats.norm(loc=np.mean(z_image_numpy), scale=np.var(z_image_numpy))

            # # percentile point, the range for the inverse cumulative distribution function:
            # bounds_for_range = distribution.cdf([0, 100])

            # # Linspace for the inverse cdf:
            # pp = np.linspace(*bounds_for_range, num=levels)

            # bins = distribution.ppf(pp)
            # print bins
        else:
            masked_reference_image = np.ma.masked_equal(reference_image, mask_value)
            masked_reference_image = np.ma.masked_less(masked_reference_image, reference_norm_range[0]*np.max(reference_image))
            masked_reference_image = np.ma.masked_greater(masked_reference_image, reference_norm_range[1]*np.max(reference_image))
            masked_image_numpy = np.ma.masked_equal(image_numpy, mask_value)
            z_image_numpy = stats.zmap(masked_image_numpy, masked_reference_image, axis=None)

            z_reference_image = stats.zscore(masked_reference_image, axis=None)

            # distribution = stats.norm(loc=np.mean(z_reference_image), scale=np.var(z_reference_image))

            # # percentile point, the range for the inverse cumulative distribution function:
            # bounds_for_range = distribution.cdf([0, 100])

            # # Linspace for the inverse cdf:
            # pp = np.linspace(*bounds_for_range, num=levels)

            # bins = distribution.ppf(pp)

            # image_range = [np.mean(z_reference_image) - np.std(z_reference_image), np.mean(z_reference_image) + np.std(z_reference_image)]
            image_range = [np.min(z_reference_image), np.max(z_reference_image)]
            bins = np.linspace(image_range[0], image_range[1], levels)

        for x in xrange(image_numpy.shape[0]):
            for y in xrange(image_numpy.shape[1]):
                for z in xrange(image_numpy.shape[2]):
                    if image_numpy[x,y,z] != mask_value:
                        image_numpy[x,y,z] = (np.abs(bins-z_image_numpy[x,y,z])).argmin() + 1

        # check_image(image_numpy, mode="maximal_slice", mask_value=mask_value)
    image_numpy[image_numpy == mask_value] = 0
    return image_numpy

def coerce_positive(image_numpy):

    """ Required by GLCM. Not sure of the implications for other algorithms.
    """

    image_min = np.min(image_numpy)
    if image_min < 0:
        image_numpy = image_numpy - image_min
    return image_numpy

def remove_islands():
    return

def erode_label(image_numpy, iterations=2, mask_value=0):
    """ For each iteration, removes all voxels not completely surrounded by
        other voxels. This might be a bit of an aggressive erosion. Also I
        would bet it is incredibly ineffecient. Also custom erosions in
        multiple dimensions look a little bit messy.
    """

    iterations = np.copy(iterations)

    if isinstance(iterations, list):
        if len(iterations) != 3:
            print 'The erosion parameter does not have enough dimensions (3). Using the first value in the eroison parameter.'
    else:
        iterations == [iterations, iterations, iterations]

    for i in xrange(max(iterations)):

        kernel_center = 0
        edges_kernel = np.zeros((3,3,3),dtype=float)
        
        if iterations[2] > 0:
            edges_kernel[1,1,0] = -1
            edges_kernel[1,1,2] = -1
            iterations[2] -= 1
            kernel_center += 2

        if iterations[1] > 0:
            edges_kernel[1,0,1] = -1
            edges_kernel[1,2,1] = -1
            iterations[1] -= 1
            kernel_center += 2

        if iterations[0] > 0:
            edges_kernel[0,1,1] = -1
            edges_kernel[2,1,1] = -1
            iterations[0] -= 1
            kernel_center += 2

        edges_kernel[1,1,1] = kernel_center

        label_numpy = np.copy(image_numpy)
        label_numpy[label_numpy != mask_value] = 1
        label_numpy[label_numpy == mask_value] = 0

        edge_image = signal.convolve(label_numpy, edges_kernel, mode='same')
        # check_image(edge_image, mode="maximal_slice")
        edge_image[edge_image < 0] = -1
        # check_image(edge_image, mode="maximal_slice")
        edge_image[np.where((edge_image <= kernel_center) & (edge_image > 0))] = -1
        # check_image(edge_image, mode="maximal_slice")
        edge_image[edge_image == 0] = 1
        # check_image(edge_image, mode="maximal_slice")
        edge_image[edge_image == -1] = 0
        # check_image(edge_image, mode="maximal_slice")
        image_numpy[edge_image == 0] = mask_value
        # check_image(image_numpy, mode="maximal_slice")


    return image_numpy

def check_image(image_numpy, second_image_numpy=[], mode="cycle", step=1, mask_value=0):

    """ A useful utiltiy for spot checks.
    """

    if second_image_numpy != []:
        for i in xrange(image_numpy.shape[0]):
            print i
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(second_image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            plt.show()
    else:
        if mode == "cycle":
            for i in xrange(image_numpy.shape[0]):
                fig = plt.figure()
                imgplot = plt.imshow(image_numpy[i,:,:], interpolation='none', aspect='auto')
                plt.show()

        if mode == "first":
            fig = plt.figure()
            imgplot = plt.imshow(image_numpy[0,:,:], interpolation='none', aspect='auto')
            plt.show()

        if mode == "maximal_slice":

            maximal = [0, np.zeros(image_numpy.shape)]

            for i in xrange(image_numpy.shape[2]):
            
                image_slice = image_numpy[:,:,i]

                test_maximal = (image_slice != mask_value).sum()

                if test_maximal >= maximal[0]:
                    maximal[0] = test_maximal
                    maximal[1] = image_slice

            fig = plt.figure()
            imgplot = plt.imshow(maximal[1], interpolation='none', aspect='auto')
            plt.show()

def check_tumor_histogram(image_numpy, second_image_numpy=[], mask_value=0, image_name = ''):

    if second_image_numpy != []:
        tumor_ROI = image_numpy[image_numpy != mask_value]
        whole_brain = second_image_numpy[second_image_numpy > (second_image_numpy.max()*0.075)]
        bins = np.linspace(second_image_numpy.max()*0.075, second_image_numpy.max(), 100)

        fig = plt.figure()

        brain_hist = fig.add_subplot(2,1,1)
        brain_hist.hist(whole_brain, bins, alpha=1, label="Whole Brain > 7.5 Percent Intensity")
        if image_name != '':
            plt.title(image_name)

        tumor_hist = fig.add_subplot(2,1,2)
        tumor_hist.hist(tumor_ROI, bins, alpha=1, label='Tumor ROI(s)', color='red')
        plt.title('Tumor ROI(s)')

        # plt.legend(loc='upper right')
        # plt.show()
        if image_name[0:3] == 'TMZ':
            plt.savefig(image_name + '.png')
        else:
            plt.savefig("Melanoma_" + image_name + '.png')
        plt.close()

    else:
        tumor_ROI = image_numpy[image_numpy != mask_value]
        plt.hist(tumor_ROI)
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        fig = plt.gcf()

        plt.show()

def save_alternate_nifti(filepath, levels, reference_image=[], method="z_score", mask_value=0):
    image = nib.load(filepath)
    image_numpy = image.get_data()
    image_affine = image.get_affine()

    if reference_image == []:
        reference_image = np.copy(image_numpy)

    image_numpy = coerce_levels(image_numpy, levels=levels, reference_image=reference_image, method=method, mask_value=mask_value)

    print 'image_transformed!'

    new_img = nib.Nifti1Image(image_numpy, image_affine)

    print 'new_image_created!'

    print 'zmapped_' + str.split(filepath, '//')[-1]

    nib.save(new_img, 'zmapped_' + str.split(filepath, '//')[-1])

def create_mosaic(image_numpy, label_numpy=[], generate_outline=True, mask_value=0, step=1, dim=2, cols=8, label_buffer=5, rotate_90=3, outfile='', flip=True):

    if label_numpy != []:

        if generate_outline:
            label_numpy = generate_label_outlines(label_numpy, dim, mask_value)

        mosaic_selections = []
        buffer_marker = 0

        for i in xrange(label_numpy.shape[dim]):
            label_slice = np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])
            if np.sum(label_slice) != 0:
                mosaic_selections += range(i-label_buffer, i+label_buffer)


        mosaic_selections = np.unique(mosaic_selections)
        mosaic_selections = mosaic_selections[mosaic_selections >= 0]
        mosaic_selections = mosaic_selections[mosaic_selections <= image_numpy.shape[dim]]


        color_range_image = [np.min(image_numpy), np.max(image_numpy)]
        color_range_label = [np.min(label_numpy), np.max(label_numpy)]

        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in xrange(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_image_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)
        mosaic_label_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)
        
        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])
            label_slice = np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])

            image_slice = np.rot90(image_slice, rotate_90)
            label_slice = np.rot90(label_slice, rotate_90)
            
            if flip:
                image_slice = np.fliplr(image_slice)
                label_slice = np.fliplr(label_slice)

            print(image_slice.size)
            if image_slice.size > 0:
                mosaic_image_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = image_slice
                mosaic_label_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = label_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        mosaic_label_numpy = np.ma.masked_where(mosaic_label_numpy == 0, mosaic_label_numpy)

        if outfile != '':
            fig = plt.figure(figsize=(mosaic_image_numpy.shape[0]/100, mosaic_image_numpy.shape[1]/100), dpi=100, frameon=False)
            plt.margins(0,0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')
            plt.imshow(mosaic_label_numpy, 'jet', vmin=color_range_label[0], vmax=color_range_label[1], interpolation='none')
            
            # try:
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=1000)
            # except:
                # print 'failure'
            # plt.show()    
            plt.clf()
            plt.close()

    else:

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]

        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in xrange(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_image_numpy = np.zeros((slice_height*np.ceil(float(image_numpy.shape[dim])/float(cols)), test_slice.shape[1]*cols), dtype=float)

        row_index = 0
        col_index = 0

        for i in xrange(image_numpy.shape[dim]):
            image_slice = np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])

            image_slice = np.rot90(image_slice, rotate_90)
            
            if flip:
                image_slice = np.fliplr(image_slice)

            mosaic_image_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = image_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        if outfile != '':
            fig = plt.figure(figsize=(mosaic_image_numpy.shape[0]/100, mosaic_image_numpy.shape[1]/100), dpi=100, frameon=False)
            plt.margins(0,0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')
            # try:
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=500)
            # except:
                # print 'failure'   
            plt.clf()
            plt.close()


def generate_label_outlines(label_numpy, dim=2, mask_value=0):

    # Will not work if someone uses 0 or <0 for a label.
    # Also this is super slow.
        
    edges_kernel = np.zeros((3,3,3),dtype=float)
    edges_kernel[1,1,1] = 4

    if dim != 2:
        edges_kernel[1,1,0] = -1
        edges_kernel[1,1,2] = -1

    if dim != 1:
        edges_kernel[1,0,1] = -1
        edges_kernel[1,2,1] = -1

    if dim != 0:
        edges_kernel[0,1,1] = -1
        edges_kernel[2,1,1] = -1
    
    outline_label_numpy = np.zeros_like(label_numpy)

    for label_number in np.unique(label_numpy):
        if label_number != mask_value:
            sublabel_numpy = np.copy(label_numpy)
            sublabel_numpy[sublabel_numpy != label_number] = 0
            edge_image = signal.convolve(sublabel_numpy, edges_kernel, mode='same')
            edge_image[sublabel_numpy != label_number] = 0
            edge_image[edge_image != 0] = label_number
            outline_label_numpy += edge_image

    return outline_label_numpy

def assert_nD(array, ndim, arg_name='image'):

    """

    This script is currently directly copied from scikit-image.

    Verify an array meets the desired ndims.
    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.
    """
    array = np.asanyarray(array)
    msg = "The parameter `%s` must be a %s-dimensional array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if not array.ndim in ndim:
        raise ValueError(msg % (arg_name, '-or-'.join([str(n) for n in ndim])))

def fill_in_convex_outline(filepath, output_file, outline_lower_threshold=[], outline_upper_threshold=[], outline_color=[], output_label_num=1, reference_nifti=[]):

    outline_upper_threshold = np.array(outline_upper_threshold)
    outline_lower_threshold = np.array(outline_lower_threshold)

    if filepath.endswith('.nii') or filepath.endswith('nii.gz'):
        return

    else:
        image_file = misc.imread(filepath)
        label_file = np.zeros_like(image_file)
        # print image_file.shape

        for row in xrange(image_file.shape[0]):
            row_section = 0
            fill_index = 0
            for col in xrange(image_file.shape[1]):
                match = False
                pixel = image_file[row, col, ...]
                
                if outline_upper_threshold != [] and outline_lower_threshold != []:
                    if all(pixel > outline_lower_threshold) and all(pixel < outline_upper_threshold):
                        match = True
                elif outline_color != []:
                    if (pixel == outline_color):
                        match = True
                else:
                    print 'Error. Please provide a valid outline color or threshold.'
                    return

                if match:
                    label_file[row, col, ...] = output_label_num                    
                    if row_section == 0:
                        row_section = 1
                    if row_section == 2:
                        row_section = 3
                        label_file[row, fill_index:col, ...] = output_label_num
                else:
                    if row_section == 1:
                        row_section = 2
                        fill_index = col
                    if row_section == 3:
                        row_section = 0

                # label_file[row, col, ...] = pixel

        label_file = binary_fill_holes(label_file[:,:,0]).astype(label_file.dtype)
        # label_file = label_file[:,:,0]

        if reference_nifti == []:
            misc.imsave(output_file, label_file)
        else:
            save_numpy_2_nifti(label_file, reference_nifti, output_file)

def replace_slice(input_nifti, output_file, input_nifti_slice, slice_num):

    input_numpy = nifti_2_numpy(input_nifti)
    input_numpy_slice = nifti_2_numpy(input_nifti_slice)
    output_numpy = np.zeros_like(input_numpy)

    output_numpy[:,:,slice_num] = np.flipud(np.rot90(input_numpy_slice[:,:]))

    save_numpy_2_nifti(output_numpy, input_nifti, output_file)


if __name__ == '__main__':
    # grab_files('C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10', 'C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10/TempFolder', '*dcm')
    np.set_printoptions(suppress=True, precision=2)
    # return_dicom_dictionary(folder="C:/Users/azb22/Documents/Scripting/Breast_MRI_Challenge/ISPY_data/ISPY1/")

    filepath = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/image_with_roi.jpg'
    output_filepath = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/image_with_roi-label-test.nii.gz'
    reference_nifti = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/7_Ax_T2_PROPELLER.nii.gz'
    output_nifti = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/7_Ax_T2_PROPELLER-label.nii.gz'

    # fill_in_convex_outline(filepath, output_filepath, outline_lower_threshold = [175,0,0], outline_upper_threshold=[300,50,50], reference_nifti=reference_nifti)
    replace_slice(reference_nifti, output_nifti, output_filepath, 10)

    # label_numpy = nifti_2_numpy('C:/Users/azb22/Documents/Scripting/TMZ Registration/TMZ_NEW/TMZ_COREGISTERED/TMZ_01-VISIT_01/TMZ_01-VISIT_01-label_r_T2_final.nii.gz')
    # image_numpy = nifti_2_numpy('C:/Users/azb22/Documents/Scripting/TMZ Registration/TMZ_NEW/TMZ_COREGISTERED/TMZ_01-VISIT_01/TMZ_01-VISIT_01-T1POST_r_T2.nii.gz')

    # for folder in glob.glob('C:\\Users\\azb22\\Documents\\Scripting\\TMZ Registration\\TMZ_NEW\\TMZ_COREGISTERED\\TMZ_0*-VISIT_0*'):
    #     if '.' in folder:
    #         continue
    #     print folder
    #     if not os.path.exists(folder + '\\Mosaic\\'):
    #         os.makedirs(folder + '\\Mosaic\\')

    #     nn_label = glob.glob(folder + '\\*label_r_T2_final*')
    #     if nn_label != []:
    #         nn_label = nn_label[0]
    #         nn_label_numpy = nifti_2_numpy(nn_label)
    #         nn_label_numpy = generate_label_outlines(nn_label_numpy)
    #         for volume in glob.glob(folder + '\\*.nii.gz'):
    #             if ('MPRAGE' in volume or 'T1POST' in volume or 'FLAIR' in volume or 'SUV' in volume) and 'label' not in volume:
    #                 image_numpy = nifti_2_numpy(volume)
    #                 filename = str.split(volume, '\\')[-1]
    #                 filename = str.split(filename, '.')[0]
    #                 filename = filename + '.png'
    #                 filename = folder +'\\Mosaic\\' + filename
    #                 print filename
    #                 try:
    #                     create_mosaic(image_numpy, nn_label_numpy, generate_outline=False, outfile=filename)
    #                 except:
    #                     print 'failure to make mosaic'
    #                 # move(filename, folder +'\\Mosaic\\' + filename)
    #     else:
    #         print 'no label!'
    # create_mosaic(image_numpy, label_numpy, outfile='test_mosaic.png')


    # path = 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/'
    # for folder in glob.glob(path + 'niiVolumes/*'):
    #     for subfolder in glob.glob(folder + '/*'):
    #         for subfile in glob.glob(subfolder + '/*'):
    #             # print subfile
    #             filename = str.split(subfile, '\\')[-1]
    #             # copy(subfile, path + filename)
    #         segmentations = glob.glob(path + 'convertedFromDicomSeg/' + '\\'.join(str.split(subfolder, '\\')[-2:]) + '\\*')
    #         print segmentations
    #         for segment in segmentations:
    #             segmentfilename = str.split(segment, '\\')[-1]
    #             copy(subfile, path + filename + '_' + segmentfilename)
    #             copy(segment, path + filename + '_' + str.split(segmentfilename, '.nii')[0] + '-label.nii')


    # path = 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/*'
    # for current_file in glob.glob(path):
    #     if '.nii' in current_file:
    #         split_file = str.split(current_file, '.nii')
    #         print current_file
    #         outfile = split_file[0] + split_file[1] + '.nii'
    #         move(current_file, outfile)

    # path = 'C:/Users/azb22/Documents/Scripting/Elizabeth_Mets_Heterogeneity/BWH_Scans'
    # for file in glob.glob(path + '/Nifti_Images*'):
    #     split_file = str.split(file, 'Nifti_Images')
    #     print split_file
    #     move(file, ''.join(split_file))

    # path = 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/niiVolumes/rider'
    # for subfolder in glob.glob(path + '/*'):
    #     for subfile in glob.glob(subfolder + '/*'):
    #         # print subfile
    #         filename = str.split(subfile, '\\')[-1]
    #         filename = str.split(filename, '.')[0]
    #         print filename
    #         # copy(subfile, path + filename)
    #     print 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/convertedFromDicomSeg/' + '\\'.join(str.split(subfolder, '/')[-1:]) + '\\*'

    #     segmentations = glob.glob('C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/convertedFromDicomSeg/' + '\\'.join(str.split(subfolder, '/')[-1:]) + '\\*')
    #     print segmentations
    #     for segment in segmentations:
    #         segmentfilename = str.split(segment, '\\')[-1]
    #         # print filename
    #         copy(subfile, 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/' + filename + '_' + segmentfilename)
    #         copy(segment, 'C:/Users/azb22/Documents/Scripting/Machine_Learning_Pipeline/qin_moistRun/' + filename + '_' + str.split(segmentfilename, '.nii')[0] + '-label.nii')
