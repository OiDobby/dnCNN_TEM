import os
import cv2
import numpy as np

def rotate_with_pbc(image, angle, final_size=512):
    """
    Rotate image with periodic boundary condition and crop to final size.
    Parameters:
        image: input 2D grayscale image
        angle: rotation angle in degrees
        final_size: output image size (final_size x final_size)
    Returns:
        Cropped rotated image with size (final_size x final_size)
    """
    h, w = image.shape
    min_len = min(h, w)
    image_cropped = image[0:min_len, 0:min_len]

    image_resized = cv2.resize(image_cropped, (final_size, final_size), interpolation=cv2.INTER_AREA)

    # Padding with wrap-around to simulate periodic boundary condition
    pad = int(np.ceil(np.sqrt(2) * final_size))
    padded = np.pad(image_resized, ((pad, pad), (pad, pad)), mode='wrap')

    # Center-based rotation
    center = (padded.shape[1] // 2, padded.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, rot_mat, (padded.shape[1], padded.shape[0]),
                             flags=cv2.INTER_LINEAR)

    # Crop to original size
    start = (rotated.shape[0] - final_size) // 2
    cropped = rotated[start:start + final_size, start:start + final_size]
    return cropped


def rotate_images_with_pbc(base_path, ori_file_dir, angle_list):
    """
    Generate rotated images (with PBC) and save them in the same folder with modified filenames.
    Parameters:
        base_path: root path
        ori_file_dir: directory containing original images (e.g., 'ori_png/')
        angle_list: list of angles in degrees to rotate (e.g., [0, 10, 20, 30])
    """
    file_dir_path = os.path.join(base_path, ori_file_dir)
    file_list = sorted(os.listdir(file_dir_path))

    print('===================================================================')
    print(f'[INFO] Rotated images will be saved in: {file_dir_path}')
    print('[INFO] Number of images:', len(file_list))
    print('[INFO] Rotation angles:', angle_list)

    for i, filename in enumerate(file_list):
        file_path = os.path.join(file_dir_path, filename)
        print(f'[{i+1}/{len(file_list)}] Processing: {filename}')

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'[WARNING] Failed to load image: {file_path}')
            continue

        for angle in angle_list:
            rot_img = rotate_with_pbc(img, angle)

            # Output file: rot{angle}_original_filename.png
            save_name = f'rot{angle}_{filename}'
            save_path = os.path.join(file_dir_path, save_name)

            cv2.imwrite(save_path, rot_img)

    print('[INFO] Rotation completed for all images.')
    print('===================================================================')


# === Example run ===
if __name__ == '__main__':
    base_path = os.getcwd()
    ori_file_dir = 'ori_png/'              # Folder with original images
    angle_list = [5, 10, 15, 20, 25, 30]   # Rotation angles in degrees

    rotate_images_with_pbc(base_path, ori_file_dir, angle_list)

