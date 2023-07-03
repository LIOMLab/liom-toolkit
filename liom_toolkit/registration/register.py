import ants


def deformably_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                               rigid_type='Rigid', deformable_type='SyN'):
    rigid, rigid_transform = rigidly_register_volume(image, mask, template, rigid_type)

    syn_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=deformable_type,
                                      initial_transform=rigid_transform['fwdtransforms'][0])
    syn = ants.apply_transforms(fixed=template, moving=image,
                                transformlist=syn_transform['fwdtransforms'])
    return syn, syn_transform, rigid_transform


def rigidly_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage, rigid_type='Rigid'):
    rigid_transform = ants.registration(fixed=template, moving=image, moving_mask=mask, type_of_transform=rigid_type)
    rigid = ants.apply_transforms(fixed=template, moving=image,
                                  transformlist=rigid_transform['fwdtransforms'])
    return rigid, rigid_transform
