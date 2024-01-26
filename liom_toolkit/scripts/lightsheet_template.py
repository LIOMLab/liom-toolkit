import ants
from tqdm import tqdm

from liom_toolkit.registration import create_template
from liom_toolkit.utils import load_zarr, load_zarr_image_from_node, load_allen_template

# List of brains used in templating
brains = [

]


def build_template_for_resolution(resolution_level=3, atlas_resolution=50, iterations=15, init_with_template=True):
    atlas_file = f"<<PATH TO ALLEN TEMPLATE>>"
    resolution_mm = atlas_resolution / 1000
    atlas_volume = load_allen_template(atlas_file, atlas_resolution, padding=False)
    atlas_volume = ants.reorient_image2(atlas_volume, "RAS")
    brain_volumes = []
    masks = []
    for brain in tqdm(brains, desc="Loading brains", leave=False, total=len(brains), unit="brain", position=1):
        zarr_file = f"<<PATH TO BRAIN FOLDER>>"
        nodes = load_zarr(zarr_file)
        image_node = nodes[0]
        mask_node = nodes[2]

        brain_volume, mask = load_volume(image_node, mask_node, resolution_level, flipped=False)
        brain_volumes.append(brain_volume)
        masks.append(mask)

        # Added flipped brains
        brain_volume, mask = load_volume(image_node, mask_node, resolution_level, flipped=True)
        brain_volumes.append(brain_volume)
        masks.append(mask)

    template = create_template(brain_volumes, masks, atlas_volume, template_resolution=resolution_mm,
                               iterations=iterations, init_with_template=init_with_template)
    template_transform = ants.registration(fixed=atlas_volume, moving=template, type_of_transform="SyN")
    template = ants.apply_transforms(fixed=atlas_volume, moving=template,
                                     transformlist=template_transform["fwdtransforms"])
    ants.image_write(template, f"lsfm_template_{atlas_resolution}_{iterations}.nii")
    return template, atlas_volume


def load_volume(image_node, mask_node, resolution_level, flipped=False):
    brain_volume = load_zarr_image_from_node(image_node, resolution_level=resolution_level)
    mask = load_zarr_image_from_node(mask_node, resolution_level=resolution_level)
    if flipped:
        direction = brain_volume.direction
        direction[0][0] = -1
        brain_volume.set_direction(direction)
        mask.set_direction(direction)
    brain_volume = ants.reorient_image2(brain_volume, "RAS")
    mask = ants.reorient_image2(mask, "RAS")
    brain_volume = brain_volume * mask
    # Fix for physical shape being reset after multiplication
    brain_volume.physical_shape = mask.physical_shape
    return brain_volume, mask


resolution_levels = [4, 3, 2]
atlas_resolutions = [100, 50, 25]

for (resolution_level, atlas_resolution) in (
        pbar := tqdm(zip(resolution_levels, atlas_resolutions), desc="Building templates", leave=True,
                     total=len(resolution_levels), unit="template", position=0)):
    pbar.set_description(f"Building template at {atlas_resolution} microns")
    build_template_for_resolution(resolution_level=resolution_level, atlas_resolution=atlas_resolution, iterations=15,
                                  init_with_template=False)
