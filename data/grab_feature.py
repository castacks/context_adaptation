import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def on_click(event, img_array):
    # Get the x and y coordinates of the click
    x = int(event.xdata)
    y = int(event.ydata)

    # Get the RGB value at the clicked point
    rgb_value = img_array[y, x, :]

    print(f"Clicked point: (x={x}, y={y})")
    print(f"RGB value at this point: {rgb_value}")

def display_image_and_select_point(image_path):
    # Open the image

    img_array = np.load('/home/matthew/physics_atv_ws/gridmap_data.npy')

    # print(img_array.shape)
    img_array = np.swapaxes(img_array, 0,2)
    img_array = np.swapaxes(img_array, 0,1)

    # Display the image in RGB
    plt.imshow(img_array[:,:,:3]/img_array[:,:,:3].max())
    plt.title('Click on the image to get RGB value')
    plt.axis('off')

    # Connect the click event to the callback function
    cid = plt.gcf().canvas.mpl_connect('button_press_event', lambda event: on_click(event, img_array))

    plt.show()

# Example usage:
image_path = 'your_image.jpg'  # Replace with your image path
display_image_and_select_point(image_path)
