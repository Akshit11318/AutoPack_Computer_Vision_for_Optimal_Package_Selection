import os
from PIL import Image, ImageDraw, ImageFont
import random
import math

# Create directories if they don't exist
os.makedirs('dataset/images', exist_ok=True)
os.makedirs('dataset/labels', exist_ok=True)

# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
REFERENCE_SIZE = 100  # 1 inch = 100 pixels
PADDING = 200  # 2 inches = 200 pixels
MIN_OBJECT_SIZE = 400  # 2 inches = 200 pixels
MAX_OBJECT_SIZE = 800  # 7 inches = 700 pixels
DIST_FROM_REF = 300  # 2 inches = 200 pixels

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_random_polygon(min_size, max_size, center_x, center_y):
    num_points = random.randint(3, 8)
    size = random.uniform(min_size, min(max_size, IMAGE_WIDTH - center_x - PADDING, IMAGE_HEIGHT - PADDING * 2))
    
    points = []
    for i in range(num_points):
        angle = math.radians(i * (360 / num_points) + random.uniform(0, 360 / num_points))
        r = size / 2 * random.uniform(0.8, 1)  # Keeping objects big
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        points.append((x, y))
    
    return points

def generate_image(dataset_id):
    # Create a new image with a white background
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw reference square (1x1 inch) in the left middle
    ref_left = PADDING
    ref_top = (IMAGE_HEIGHT - REFERENCE_SIZE) // 2
    draw.rectangle([ref_left, ref_top, 
                    ref_left + REFERENCE_SIZE, ref_top + REFERENCE_SIZE], 
                   fill='black')
    
    # Calculate object center (2 inches to the right of the reference square)
    object_center_x = ref_left + REFERENCE_SIZE + DIST_FROM_REF
    object_center_y = IMAGE_HEIGHT // 2
    
    # Generate and draw random polygon
    polygon_points = generate_random_polygon(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE, object_center_x, object_center_y)
    draw.polygon(polygon_points, fill=random_color(), outline='black')
    
    # Calculate object dimensions
    min_x = min(p[0] for p in polygon_points)
    max_x = max(p[0] for p in polygon_points)
    min_y = min(p[1] for p in polygon_points)
    max_y = max(p[1] for p in polygon_points)
    width = (max_x - min_x) / REFERENCE_SIZE
    height = (max_y - min_y) / REFERENCE_SIZE
    
    
    '''# Add labels
    font = ImageFont.load_default()
    draw.text((IMAGE_WIDTH - 150, 10), f"Dataset ID: {dataset_id}", fill='black', font=font)
    draw.text((IMAGE_WIDTH - 150, 40), f"W: {width:.2f}\"", fill='black', font=font)
    draw.text((IMAGE_WIDTH - 150, 70), f"H: {height:.2f}\"", fill='black', font=font)
    '''
    return img, width, height

def main():
    num_images = 1000  # Change this to generate more or fewer images
    
    for i in range(num_images):
        dataset_id = f"{i:04d}"
        img, width, height = generate_image(dataset_id)
        
        # Save image
        img.save(f'dataset/images/{dataset_id}.png')
        
        # Save label
        with open(f'dataset/labels/predicted_{dataset_id}.txt', 'w') as f:
            f.write(f"Width: {width:.2f} inches\n")
            f.write(f"Height: {height:.2f} inches\n")
        
        print(f"Generated image and label for dataset ID: {dataset_id}")

if __name__ == "__main__":
    main()
