
import os
import pandas as pd
from PIL import Image, ImageDraw

# Define image parameters
IMG_SIZE = 64
NUM_IMAGES_PER_CLASS = 50
CLASSES = ["circle", "square", "triangle"]
OUTPUT_DIR = "data/raw"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

def main():
    """
    Generates a synthetic dataset of geometric shapes.
    """
    print("Generating synthetic image data...")

    # Create the output directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    data = []

    for class_name in CLASSES:
        for i in range(NUM_IMAGES_PER_CLASS):
            # Create a new black image
            img = Image.new('L', (IMG_SIZE, IMG_SIZE), 'black')
            draw = ImageDraw.Draw(img)

            # Draw the shape
            if class_name == "circle":
                draw.ellipse([(10, 10), (IMG_SIZE - 10, IMG_SIZE - 10)], fill='white')
            elif class_name == "square":
                draw.rectangle([(10, 10), (IMG_SIZE - 10, IMG_SIZE - 10)], fill='white')
            elif class_name == "triangle":
                draw.polygon([(IMG_SIZE // 2, 10), (10, IMG_SIZE - 10), (IMG_SIZE - 10, IMG_SIZE - 10)], fill='white')

            # Save the image
            img_filename = f"{class_name}_{i}.png"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            img.save(img_path)

            data.append({"filepath": img_path, "label": class_name})

    # Create and save the CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    df.to_csv(csv_path, index=False)

    print(f"Successfully generated {len(data)} images and saved labels to {csv_path}")

if __name__ == "__main__":
    main()
