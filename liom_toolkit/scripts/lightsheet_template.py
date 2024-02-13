import ants
from tqdm import tqdm

from liom_toolkit.registration import create_template
from liom_toolkit.utils import load_zarr, load_allen_template, load_zarr_image_from_node, segment_3d_brain

# Brains to use in registration
brains = [

]

# The list can be used to save the mirrored versions if required.
brain_names = brains


def build_template_for_resolution(template_name, resolution_level=3, template_resolution=50, iterations=15,
                                  init_with_template=False,
                                  register_to_template=False, flipped_brains=False, wavelength="647nm"):
    resolution_mm = template_resolution / 1000
    atlas_file = f"/data/templates/allen/average_template_{template_resolution}.nrrd"
    atlas_volume = load_allen_template(atlas_file, template_resolution, padding=False)
    atlas_volume = ants.reorient_image2(atlas_volume, "RAS")
    brain_volumes = []
    masks = []
    for brain in tqdm(brains, desc="Loading brains", leave=False, total=len(brains), unit="brain", position=1):
        zarr_file = f"/data/LSFM/{brain}/{wavelength}.zarr"
        nodes = load_zarr(zarr_file)
        image_node = nodes[0]
        mask_node = nodes[2]

        brain_volume, mask = load_volume(image_node, mask_node, resolution_level, flipped=False)
        brain_volumes.append(brain_volume)
        masks.append(mask)

        # Added flipped brains
        if flipped_brains:
            brain_volume, mask = load_volume(image_node, mask_node, resolution_level, flipped=True)
            brain_volumes.append(brain_volume)
            masks.append(mask)

    if init_with_template:
        template = create_template(brain_volumes, masks, brain_names, atlas_volume,
                                   template_resolution=resolution_mm, iterations=iterations,
                                   pre_registration_type="Rigid")
    else:
        template = create_template(brain_volumes, masks, brain_names, atlas_volume,
                                   template_resolution=resolution_mm, iterations=iterations,
                                   init_with_template=init_with_template, pre_registration_type="Rigid")
    if register_to_template:
        template_transform = ants.registration(fixed=atlas_volume, moving=template, type_of_transform="SyN")
        template = ants.apply_transforms(fixed=atlas_volume, moving=template,
                                         transformlist=template_transform["fwdtransforms"])
    # Mask template to remove noise
    template_mask = segment_3d_brain(template)
    new_template = template * template_mask
    # Apply properties after multiplication
    new_template.set_direction(template.direction)
    new_template.set_spacing(template.spacing)
    new_template.set_origin(template.origin)

    ants.image_write(new_template, f"templates/{template_name}_{template_resolution}_{iterations}.nii")

    return template, atlas_volume


def load_volume(image_node, mask_node, resolution_level, flipped=False):
    brain_volume = load_zarr_image_from_node(image_node, resolution_level=resolution_level)
    mask = load_zarr_image_from_node(mask_node, resolution_level=resolution_level)
    brain_volume = brain_volume * mask
    if flipped:
        direction = brain_volume.direction
        direction[0][0] = -1
        brain_volume.set_direction(direction)
        mask.set_direction(direction)
    brain_volume = ants.reorient_image2(brain_volume, "RAS")
    mask = ants.reorient_image2(mask, "RAS")
    # Fix for physical shape being reset after multiplication
    brain_volume.physical_shape = mask.physical_shape
    return brain_volume, mask


resolution_levels = [2, 3]
atlas_resolutions = [25, 50]

for (resolution_level, atlas_resolution) in (
        pbar := tqdm(zip(resolution_levels, atlas_resolutions), desc="Building templates", leave=True,
                     total=len(resolution_levels), unit="template", position=0)):
    pbar.set_description(f"Building template at {atlas_resolution} microns")
    build_template_for_resolution("", resolution_level=resolution_level,
                                  template_resolution=atlas_resolution, iterations=15,
                                  init_with_template=False, register_to_template=False, flipped_brains=False,
                                  wavelength="647nm")
