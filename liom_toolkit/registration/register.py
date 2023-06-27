import ants


def deformable_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage,
                               rigid_type='Rigid', deformable_type='SyN'):
    rigid, rigid_transform = rigid_register_volume(image, mask, template, rigid_type)
    mask_rigid = ants.apply_transforms(fixed=template, moving=mask,
                                       transformlist=rigid_transform['fwdtransforms'])

    syn_transform = ants.registration(fixed=template, moving=rigid, mask=mask_rigid, type_of_transform=deformable_type)
    syn = ants.apply_transforms(fixed=template, moving=rigid,
                                transformlist=syn_transform['fwdtransforms'])
    return syn, syn_transform


def rigid_register_volume(image: ants.ANTsImage, mask: ants.ANTsImage, template: ants.ANTsImage, rigid_type='Rigid'):
    rigid_transform = ants.registration(fixed=template, moving=image, mask=mask, type_of_transform=rigid_type)
    rigid = ants.apply_transforms(fixed=template, moving=image,
                                  transformlist=rigid_transform['fwdtransforms'])
    return rigid, rigid_transform
