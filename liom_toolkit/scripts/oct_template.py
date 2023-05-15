import ants
from tqdm import tqdm

from liom_toolkit.registration import segment_3d_brain, create_template

# For brains in RSP direction
brains = [

]
# For brains in RAS direction
ras_brains = [

]
resolution = 25
resolution_mm = resolution / 1000

# Load Allen atlas template
atlas_file = f"<<PATH TO ALLEN TEMPLATE>>"
atlas_volume = ants.image_read(atlas_file)
atlas_volume.set_direction([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
atlas_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
atlas_volume = ants.reorient_image2(atlas_volume, "RAS")

# Register the base to the atlas template


# Load OCT volumes
oct_volumes = []
for brain in brains:
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_direction([[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")
    oct_volumes.append(oct_volume)

    # Add mirrored volume
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_direction([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")
    oct_volumes.append(oct_volume)

for brain in ras_brains:
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")
    oct_volumes.append(oct_volume)

    # Add mirrored volume
    oct_volume = ants.image_read(f"oct_data/{brain}.nii")
    oct_volume.set_direction([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    oct_volume.set_spacing([resolution_mm, resolution_mm, resolution_mm])
    oct_volume = ants.reorient_image2(oct_volume, "RAS")
    oct_volumes.append(oct_volume)

# Create masks for all volumes
oct_masks = []
for oct_volume in tqdm(oct_volumes, desc="Creating masks", total=len(oct_volumes), unit="volumes"):
    oct_mask = segment_3d_brain(oct_volume)
    oct_mask = oct_mask.astype("uint8") * 255
    oct_mask = ants.from_numpy(oct_mask)
    oct_mask.set_direction(oct_volume.direction)
    oct_mask.set_spacing(oct_volume.spacing)
    oct_mask.set_origin(oct_volume.origin)
    oct_masks.append(oct_mask)

# Apply masks
oct_volumes = [oct_volume * oct_mask for oct_volume, oct_mask in zip(oct_volumes, oct_masks)]

# Create template
template = create_template(oct_volumes, oct_masks, atlas_volume, resolution_mm, 15)

# Register template to atlas
template_transform = ants.registration(fixed=atlas_volume, moving=template, type_of_transform="SyN")
template = ants.apply_transforms(fixed=atlas_volume, moving=template, transformlist=template_transform["fwdtransforms"])
ants.image_write(template, f"templates/oct_average_template_{25}um.nii")
