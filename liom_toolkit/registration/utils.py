import os

import ants
import ants.core.ants_image as iio
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from ants import utils


def convert_allen_nrrd_to_ants(volume: np.ndarray, resolution: float) -> ants.ANTsImage:
    """
    Convert a nrrd file form the Allen reference spaces to an ants image. The returned image will be in RAS+ orientation.

    :param volume: The already loaded nrrd file.
    :type volume: np.ndarray
    :param resolution: The resolution of the nrrd file in millimeters.
    :type resolution: float
    :return: The converted image.
    :rtype: ants.ANTsImage
    """
    # Set axis to RAS
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    # Convert to ants image and set direction and spacing
    volume = ants.from_numpy(volume.astype("uint32"))
    volume.set_direction([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    volume.set_spacing([resolution, resolution, resolution])

    return volume


def construct_reference_space_cache(resolution: int = 25) -> ReferenceSpaceCache:
    """
    Construct a reference space cache for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :return: The reference space cache.
    :rtype: ReferenceSpaceCache
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = ReferenceSpaceCache(resolution=resolution, reference_space_key="annotation/ccf_2017")

    return rsc


def download_allen_template(data_dir: str, resolution: int = 25, keep_nrrd: bool = False,
                            rsc: ReferenceSpaceCache = None) -> ants.ANTsImage:
    """
    Download the allen mouse brain template in RAS+ orientation.

    :param data_dir: The directory to save the template to.
    :type data_dir: str
    :param resolution: The template resolution in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :type keep_nrrd: bool
    :param rsc: The reference space cache to use. If None, a new one will be constructed.
    :type rsc: ReferenceSpaceCache
    :return: The template as an ants image.
    :rtype: ants.ANTsImage
    """
    # Check the resolution
    assert int(resolution) in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # filename
    nrrd_file = f"{data_dir}/allen_template_{resolution}.nrrd"

    # Downloading the template
    if rsc is None:
        rsc = construct_reference_space_cache(resolution=resolution)
    vol, metadata = rsc.get_template_volume(file_name=str(nrrd_file))

    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image


def download_allen_atlas(data_dir: str, resolution: int = 25, keep_nrrd: bool = False,
                         rsc: ReferenceSpaceCache = None) -> ants.ANTsImage:
    """
    Download the allen mouse brain atlas and reorient it to RAS+.

    :param data_dir: The directory to save the atlas to.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :param resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :param keep_nrrd: bool
    :param rsc: The reference space cache to use. If None, a new one will be constructed.
    :param rsc: ReferenceSpaceCache
    :return: The atlas as an ants image.
    :rtype: ants.ANTsImage
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"
    # Download resolution is 10 micron to fix wrong region labels

    # Temporary filename
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"

    # Downloading the annotation (if not already downloaded)
    if rsc is None:
        rsc = construct_reference_space_cache(resolution=resolution)
    vol, metadata = rsc.get_annotation_volume(str(nrrd_file))

    # Convert to ants image
    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image


def apply_transforms(fixed, moving, transformlist,
                     interpolator='linear', imagetype=0,
                     whichtoinvert=None, compose=None,
                     defaultvalue=0, verbose=False, **kwargs):
    """
    FIXME: Function temporarily copied from ANTsPy. Will be removed once the issue with applying the transforms with double precision is resolved.

    Apply a transform list to map an image from one domain to another.
    In image registration, one computes mappings between (usually) pairs
    of images. These transforms are often a sequence of increasingly
    complex maps, e.g. from translation, to rigid, to affine to deformation.
    The list of such transforms is passed to this function to interpolate one
    image domain into the next image domain, as below. The order matters
    strongly and the user is advised to familiarize with the standards
    established in examples.

    ANTsR function: `antsApplyTransforms`

    Arguments
    ---------
    fixed : ANTsImage
        fixed image defining domain into which the moving image is transformed.

    moving : AntsImage
        moving image to be mapped to fixed space.

    transformlist : list of strings
        list of transforms generated by ants.registration where each transform is a filename.

    interpolator : string
        Choice of interpolator. Supports partial matching.
            linear
            nearestNeighbor
            multiLabel for label images (deprecated, prefer genericLabel)
            gaussian
            bSpline
            cosineWindowedSinc
            welchWindowedSinc
            hammingWindowedSinc
            lanczosWindowedSinc
            genericLabel use this for label images

    imagetype : integer
        choose 0/1/2/3 mapping to scalar/vector/tensor/time-series

    whichtoinvert : list of booleans (optional)
        Must be same length as transformlist.
        whichtoinvert[i] is True if transformlist[i] is a matrix,
        and the matrix should be inverted. If transformlist[i] is a
        warp field, whichtoinvert[i] must be False.
        If the transform list is a matrix followed by a warp field,
        whichtoinvert defaults to (True,False). Otherwise it defaults
        to [False]*len(transformlist)).

    compose : string (optional)
        if it is a string pointing to a valid file location,
        this will force the function to return a composite transformation filename.

    defaultvalue : scalar
        Default voxel value for mappings outside the image domain.

    verbose : boolean
        print command and run verbose application of transform.

    kwargs : keyword arguments
        extra parameters

    Returns
    -------
    ANTsImage or string (transformation filename)

    Example
    -------
    >>> import ants
    >>> fixed = ants.image_read( ants.get_ants_data('r16') )
    >>> moving = ants.image_read( ants.get_ants_data('r64') )
    >>> fixed = ants.resample_image(fixed, (64,64), 1, 0)
    >>> moving = ants.resample_image(moving, (64,64), 1, 0)
    >>> mytx = ants.registration(fixed=fixed , moving=moving ,
                                 type_of_transform = 'SyN' )
    >>> mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving,
                                               transformlist=mytx['fwdtransforms'] )
    """

    if not isinstance(transformlist, (tuple, list)) and (transformlist is not None):
        transformlist = [transformlist]

    accepted_interpolators = {"linear", "nearestNeighbor", "multiLabel", "gaussian",
                              "bSpline", "cosineWindowedSinc", "welchWindowedSinc",
                              "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel"}

    if interpolator not in accepted_interpolators:
        raise ValueError('interpolator not supported - see %s' % accepted_interpolators)

    args = [fixed, moving, transformlist, interpolator]

    if not isinstance(fixed, str):
        if isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage):
            for tl_path in transformlist:
                if not os.path.exists(tl_path):
                    raise Exception('Transform %s does not exist' % tl_path)

            inpixeltype = fixed.pixeltype
            fixed = fixed.clone('double')
            moving = moving.clone('double')
            warpedmovout = moving.clone('double')
            f = fixed
            m = moving
            if (moving.dimension == 4) and (fixed.dimension == 3) and (imagetype == 0):
                raise Exception('Set imagetype 3 to transform time series images.')

            wmo = warpedmovout
            mytx = []
            if whichtoinvert is None or (
                    isinstance(whichtoinvert, (tuple, list)) and (sum([w is not None for w in whichtoinvert]) == 0)):
                if (len(transformlist) == 2) and ('.mat' in transformlist[0]) and ('.mat' not in transformlist[1]):
                    whichtoinvert = (True, False)
                else:
                    whichtoinvert = tuple([False] * len(transformlist))

            if len(whichtoinvert) != len(transformlist):
                raise ValueError('Transform list and inversion list must be the same length')

            for i in range(len(transformlist)):
                ismat = False
                if '.mat' in transformlist[i]:
                    ismat = True
                if whichtoinvert[i] and (not ismat):
                    raise ValueError(
                        'Cannot invert transform %i (%s) because it is not a matrix' % (i, transformlist[i]))
                if whichtoinvert[i]:
                    mytx = mytx + ['-t', '[%s,1]' % (transformlist[i])]
                else:
                    mytx = mytx + ['-t', transformlist[i]]

            if compose is None:
                args = ['-d', fixed.dimension,
                        '-i', m,
                        '-o', wmo,
                        '-r', f,
                        '-n', interpolator]
                args = args + mytx
            if compose:
                tfn = '%scomptx.nii.gz' % compose if not compose.endswith('.h5') else compose
            else:
                tfn = 'NA'
            if compose is not None:
                mycompo = '[%s,1]' % tfn
                args = ['-d', fixed.dimension,
                        '-i', m,
                        '-o', mycompo,
                        '-r', f,
                        '-n', interpolator]
                args = args + mytx

            myargs = utils._int_antsProcessArguments(args)

            myverb = int(verbose)
            if verbose:
                print(myargs)

            processed_args = myargs + ['-z', str(1), '-v', str(myverb), str(1), '-e', str(imagetype), '-f',
                                       str(defaultvalue)]
            libfn = utils.get_lib_fn('antsApplyTransforms')
            libfn(processed_args)

            if compose is None:
                return warpedmovout.clone(inpixeltype)
            else:
                if os.path.exists(tfn):
                    return tfn
                else:
                    return None

        else:
            return 1
    else:
        args = args + ['-z', 1, '--float', 1, '-e', imagetype, '-f', defaultvalue]
        processed_args = utils._int_antsProcessArguments(args)
        libfn = utils.get_lib_fn('antsApplyTransforms')
        libfn(processed_args)
