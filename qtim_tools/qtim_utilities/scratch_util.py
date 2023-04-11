""" This is a scratch utilty folder, for idiosyncratic functions
    that don't fit anywhere else but might want to be kept later.
"""
from nifti_utils import *
import nibabel

def save_alternate_nifti(filepath, levels, reference_image=[], method="z_score", mask_value=0):

    """ Consider for deletion, or scrap pile
    """

    image = nib.load(filepath)
    image_numpy = image.get_fdata
    image_affine = image.affine

    if reference_image == []:
        reference_image = np.copy(image_numpy)

    image_numpy = coerce_levels(image_numpy, levels=levels, reference_image=reference_image, method=method, mask_value=mask_value)

    print('image_transformed!')

    new_img = nib.Nifti1Image(image_numpy, image_affine)

    print('new_image_created!')

    print('zmapped_' + str.split(filepath, '//')[-1])

    nib.save(new_img, 'zmapped_' + str.split(filepath, '//')[-1])
