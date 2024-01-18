import cv2
import os
import numpy as np


def augment_data(dataset_path):
    """
    Augment images in the specified dataset path and save the augmented images in a temporary output folder.

    Parameters:
    - dataset_path (str): Path to the dataset containing images.
    """
    # Specify the temporary output folder
    output_path = "./temp/"
    os.makedirs(output_path, exist_ok=True)

    # List all image files in the dataset path
    image_files = [f for f in os.listdir(
        dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)

        # Read the image
        original_image = cv2.imread(image_path)

        # Apply augmentations and save images
        # Rotation
        angle = np.random.randint(-10, 10)
        rotated_image = cv2.warpAffine(original_image, cv2.getRotationMatrix2D(
            (original_image.shape[1] // 2, original_image.shape[0] // 2), angle, 1.0), (original_image.shape[1], original_image.shape[0]))

        # Scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_image = cv2.resize(
            original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Translation
        tx, ty = np.random.randint(-20, 20, 2)
        translated_image = cv2.warpAffine(original_image, np.float32(
            [[1, 0, tx], [0, 1, ty]]), (original_image.shape[1], original_image.shape[0]))

        # Flipping
        flip_direction = np.random.randint(0, 2)
        flipped_image = cv2.flip(original_image, flip_direction)

        # Brightness adjustment
        brightness = np.random.uniform(0.5, 1.5)
        brightened_image = cv2.convertScaleAbs(
            original_image, alpha=brightness)

        magnitude = np.random.randint(1, 10)
        rows, cols, _ = brightened_image.shape
        dx, dy = np.random.randint(-magnitude, magnitude, 2)
        random_points = np.float32(
            [[0, 0], [cols, 0], [0, rows], [cols, rows]])
        destination_points = np.float32(
            [[0, 0], [cols + dx, dy], [dx, rows + dy], [cols + dx, rows + dy]])
        matrix = cv2.getPerspectiveTransform(random_points, destination_points)
        warped_image = cv2.warpPerspective(
            brightened_image, matrix, (cols, rows))

        # Save augmented images
        cv2.imwrite(os.path.join(
            output_path, f"rotated__{image_file}"), rotated_image)
        cv2.imwrite(os.path.join(
            output_path, f"scaled__{image_file}"), scaled_image)
        cv2.imwrite(os.path.join(
            output_path, f"translated__{image_file}"), translated_image)
        cv2.imwrite(os.path.join(
            output_path, f"flipped__{image_file}"), flipped_image)
        cv2.imwrite(os.path.join(
            output_path, f"brightened__{image_file}"), brightened_image)
        cv2.imwrite(os.path.join(
            output_path, f"warped__{image_file}"), warped_image)


def reaugment_data(output_path):
    """
    Re-augment images from the temporary folder and save the re-augmented images in the specified output path.

    Parameters:
    - output_path (str): Path to save the re-augmented images.
    """
    dataset_path = "./temp/"

    # List all image files in the dataset path
    image_files = [f for f in os.listdir(
        dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)

        # Read the image
        original_image = cv2.imread(image_path)

        # Apply augmentations and save images
        # Rotation
        angle = np.random.randint(-10, 10)
        rotated_image = cv2.warpAffine(original_image, cv2.getRotationMatrix2D(
            (original_image.shape[1] // 2, original_image.shape[0] // 2), angle, 1.0), (original_image.shape[1], original_image.shape[0]))

        # Scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_image = cv2.resize(
            rotated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Translation
        tx, ty = np.random.randint(-20, 20, 2)
        translated_image = cv2.warpAffine(scaled_image, np.float32(
            [[1, 0, tx], [0, 1, ty]]), (scaled_image.shape[1], scaled_image.shape[0]))

        # Flipping
        flip_direction = np.random.randint(0, 2)
        flipped_image = cv2.flip(translated_image, flip_direction)

        # Brightness adjustment
        brightness = np.random.uniform(0.5, 1.5)
        brightened_image = cv2.convertScaleAbs(flipped_image, alpha=brightness)

        # Save augmented images
        cv2.imwrite(os.path.join(
            output_path, f"rotated__{image_file}"), rotated_image)
        cv2.imwrite(os.path.join(
            output_path, f"scaled__{image_file}"), scaled_image)
        cv2.imwrite(os.path.join(
            output_path, f"translated__{image_file}"), translated_image)
        cv2.imwrite(os.path.join(
            output_path, f"flipped__{image_file}"), flipped_image)
        cv2.imwrite(os.path.join(
            output_path, f"brightened__{image_file}"), brightened_image)

    # Remove the temporary folder
    os.system("rm -rf ./temp/")


def main():
    """
    Main function to demonstrate image augmentation and re-augmentation.
    """
    dataset_path = "./data/"
    output_path = "./augmented_data/"
    os.makedirs(output_path, exist_ok=True)

    augment_data(dataset_path)
    reaugment_data(output_path)


if __name__ == "__main__":
    main()
