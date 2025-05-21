"""
Generate placeholder icons for the Graphiti Browser Agent extension.
This script creates simple colored squares with the letter 'G' for different icon sizes.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Create icons directory if it doesn't exist
os.makedirs("icons", exist_ok=True)

# Icon sizes
sizes = [16, 48, 128]

# Colors
background_color = (74, 108, 247)  # #4a6cf7
text_color = (255, 255, 255)  # white

for size in sizes:
    # Create a new image with the specified size
    img = Image.new('RGB', (size, size), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, or use default if not available
    try:
        # Adjust font size based on icon size
        font_size = int(size * 0.6)
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the letter 'G' in the center
    text = "G"
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else font.getsize(text)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    draw.text(position, text, fill=text_color, font=font)
    
    # Save the image
    img.save(f"icons/icon{size}.png")
    
    print(f"Created icon{size}.png")

print("Icon generation complete!")