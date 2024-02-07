import ants


def deformably_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                               rigid_type='Rigid', deformable_type='SyN'):
    """
    Register an image to a template using a rigid registration followed by a deformable registration.

    :param image: The image to register
    :param mask: The mask to use in registration
    :param template: The template to register to
    :param rigid_type: The type of rigid registration to use
    :param deformable_type: The type of deformable registration to use
    :return: The registered image, the transform from the rigid registration,
            and the transform from the deformable registration
    """
    rigid, rigid_transform = rigidly_register_volume(image, mask, template, rigid_type)

    syn_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=deformable_type,
                                      initial_transform=rigid_transform['fwdtransforms'][0])
    syn = ants.apply_transforms(fixed=template, moving=image,
                                transformlist=syn_transform['fwdtransforms'])
    return syn, syn_transform, rigid_transform


def rigidly_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage, rigid_type='Rigid'):
    """
    Register an image to a template using a rigid registration.

    :param image: The image to register
    :param mask: The mask to use in registration
    :param template: The template to register to
    :param rigid_type: The type of rigid registration to use
    :return: The registered image and the transform from the rigid registration
    """
    rigid_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=rigid_type)
    rigid = ants.apply_transforms(fixed=template, moving=image,
                                  transformlist=rigid_transform['fwdtransforms'])
    return rigid, rigid_transform
