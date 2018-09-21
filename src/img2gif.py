import imageio
import os 

png_dir = '../img/DNN_hor'
images = []
print(sorted(os.listdir(png_dir)))
for filename in sorted(os.listdir(png_dir)):
    if filename.endswith('.png'):
        file_path = os.path. join(png_dir, filename)
        images.append(imageio.imread(file_path))
imageio.mimsave('./DNN_hor.gif', images)

