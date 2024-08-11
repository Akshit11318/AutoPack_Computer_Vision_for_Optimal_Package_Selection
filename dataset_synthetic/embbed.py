from PIL import Image, ImageDraw, ImageFont
import os

# Directories
images_folder = 'predict'
texts_folder = 'dataset/labels'
final_dir = 'embedd'

# Ensure the final directory exists
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

# Load font
font_size = 20
try:
    # Try to load a default font
    font = ImageFont.load_default()
except IOError:
    print("Default font loading failed. Please ensure Pillow is installed correctly.")
    font = ImageFont.load_default()

# Get a list of all images
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Process each image and label
for image_file in image_files:
    print(f"Processing file: {image_file}")

    # Load image
    image_path = os.path.join(images_folder, image_file)
    try:
        image = Image.open(image_path)
    except IOError as e:
        print(f"Error opening image {image_file}: {e}")
        continue

    draw = ImageDraw.Draw(image)

    # Determine the corresponding label file
    base_name = os.path.splitext(image_file)[0]  # Remove file extension
    label_file = f"{base_name}.txt"  # Use the base name to find the label file
    label_path = os.path.join(texts_folder, label_file)

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            text = file.read().strip()
        
        print(f"Label for {image_file}: {text}")

        # Get image dimensions
        image_width, image_height = image.size

        # Calculate text size and position
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            print("Error calculating text size. Using default font.")
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        text_x = image_width - text_width - 10  # 10 pixels from the right
        text_y = 10  # 10 pixels from the top

        # Print debug information
        print(f"Text bounding box: {bbox}")
        print(f"Text position: ({text_x}, {text_y})")

        # Overlay text onto image
        draw.text((text_x, text_y), text, font=font, fill="black")

        # Save the new image
        final_image_path = os.path.join(final_dir, image_file)
        try:
            image.save(final_image_path)
            print(f"Saved image to {final_image_path}")
        except IOError as e:
            print(f"Error saving image {image_file}: {e}")
            continue
    else:
        print(f"Label file {label_file} does not exist. Skipping.")

print("Processing complete. Check the 'embedd' directory for the output images.")

