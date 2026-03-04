import numpy as np
from PIL import Image


def add_gaussian_noise(image_path: str, output_path: str, mean: float = 0, std: float = 25) -> None:
    """
    Add Gaussian noise to an image and save the result.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the noisy image
        mean: Mean of the Gaussian noise (default: 0)
        std: Standard deviation of the Gaussian noise (default: 25)
    """
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.float64)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, std, image_array.shape)
    
    # Add noise to the image
    noisy_image_array = image_array + noise
    
    # Clip values to valid range [0, 255] and convert back to uint8
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image and save
    noisy_image = Image.fromarray(noisy_image_array)
    noisy_image.save(output_path)
    
    print(f"Noisy image saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    input_image_path = "/mnt/data1/xiongyizhe/dataset/imnet-ood/imagenet-c/gaussian_noise/5/n01440764/ILSVRC2012_val_00000293.JPEG"  # Replace with your image path
    output_image_path = "noisy_image.png"  # Replace with desired output path
    
    add_gaussian_noise(input_image_path, output_image_path, mean=0, std=125)