# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:30:29 2023

@author: Rafael
"""

import glob
from PIL import Image, ImageDraw

# Create a blank image with the dimensions of the grid
grid_width = 4000 # specify the width of the grid
grid_height = 6000 # specify the height of the grid
grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

var = 'result_qm'
figure_files = glob.glob('figures/'+var+'*.png')
order = [3, 0, 2, 5, 1, 4]
figure_files = [figure_files[i] for i in order]

for i, figure_file in enumerate(figure_files):
    # Load the figure image
    figure_image = Image.open(figure_file).resize((grid_width // 2, grid_height // 3))

    # Calculate the x and y coordinates of the figure on the grid
    row = i // 2
    col = i % 2
    x = col * (grid_width // 2)
    y = row * (grid_height // 3)
    # print(x)
    # print(y)
    
    # Paste the figure onto the grid image
    grid_image.paste(figure_image, (x, y))

# Save the grid image
grid_image.save(var+'_all.png')


