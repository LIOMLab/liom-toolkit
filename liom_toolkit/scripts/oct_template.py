import ants
from tqdm import tqdm

from liom_toolkit.registration import create_template
from liom_toolkit.segmentation.volume_segmentation import segment_3d_brain


def load_allen_template(resolution, resolution_mm):
    # Load Allen atlas template
    atlas_file = f"<<PATH TO ALLEN TEMPLATE>>"
    atlas_volume = ants.image_read(atlas_file)
    atlas_volume.set_direction([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
    atlas_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    atlas_volume = ants.reorient_image2(atlas_volume, "RAS")
    return atlas_volume


def load_brain(brain, resolution_mm, mirrored=False):
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    if mirrored:
        oct_volume.set_direction([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    else:
        oct_volume.set_direction([[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")
    return oct_volume


def load_brains(brains, resolution_mm):
    oct_volumes = []
    for brain in brains:
        oct_volume = load_brain(brain, resolution_mm)
        oct_volumes.append(oct_volume)

        # Add mirrored volume
        oct_volume = load_brain(brain, resolution_mm, mirrored=True)
        oct_volumes.append(oct_volume)
    return oct_volumes


def load_ras_brain(brain, resolution_mm, mirrored=False):
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    if mirrored:
        oct_volume.set_direction([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")

    return oct_volume


def load_ras_brains(brains, resolution_mm):
    oct_volumes = []
    for brain in brains:
        oct_volume = load_ras_brain(brain, resolution_mm)
        oct_volumes.append(oct_volume)

        # Add mirrored volume
        oct_volume = load_ras_brain(brain, resolution_mm, mirrored=True)
        oct_volumes.append(oct_volume)
    return oct_volumes


def create_mask(volume):
    oct_mask = segment_3d_brain(volume)
    oct_mask = oct_mask.astype("uint8")
    oct_mask = ants.from_numpy(oct_mask)
    oct_mask.set_direction(volume.direction)
    oct_mask.set_spacing(volume.spacing)
    oct_mask.set_origin(volume.origin)
    return oct_mask


def create_masks(volumes):
    oct_masks = []
    for volume in tqdm(volumes, desc="Creating masks", total=len(volumes), unit="volumes"):
        oct_mask = create_mask(volume)
        oct_masks.append(oct_mask)

    return oct_masks


def build_template(brains, ras_brains, resolution_mm, iterations):
    # Register the base to the atlas template

    oct_volumes = load_brains(brains, resolution_mm)

    # Load OCT volumes

    ras_volumes = load_ras_brains(ras_brains, resolution_mm)
    oct_volumes.extend(ras_volumes)

    # Create masks for all volumes
    oct_masks = create_masks(oct_volumes)

    # Apply masks
    oct_volumes = [oct_volume * oct_mask for oct_volume, oct_mask in zip(oct_volumes, oct_masks)]

    # Create template
    template = create_template(oct_volumes, oct_masks, atlas_volume, resolution_mm, iterations)
    return template


# For brains in RSP direction
brains = [

]
# For brains in RAS direction
ras_brains = [

]
resolution = 25
resolution_mm = resolution / 1000
iterations = 15

atlas_volume = load_allen_template(resolution, resolution_mm)
template = build_template(brains, ras_brains, resolution, iterations)

# Register template to atlas
template_transform = ants.registration(fixed=atlas_volume, moving=template, type_of_transform="SyN")
template = ants.apply_transforms(fixed=atlas_volume, moving=template, transformlist=template_transform["fwdtransforms"])
ants.image_write(template, f"templates/oct_average_template_{resolution}um.nii")
