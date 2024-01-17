import os
import PIL
from PIL import Image

def imgBicubicScale(sor_image, tar_image, scale):
    img = Image.open(sor_image)
    width, height = img.size
    height_new = int(height*scale)
    width_new = int(width*scale)
    if width / height >= width_new / height_new:
        img_new = img.resize((width_new, int(height * width_new / width)), PIL.Image.BICUBIC)
    else:
        img_new = img.resize((int(width * height_new / height), height_new), PIL.Image.BICUBIC)
    img_new.save(tar_image)
    print('resize to %s' %(tar_image))


def folder_images_resize(source_folder, target_folder, scale, output_ext='.png'):
    for root, dirs, files in sorted(os.walk(source_folder, topdown=False)):
        for name in files:
            if os.path.splitext(name)[1] == output_ext:
                sou_path_file = os.path.join(root, name)
                # print('sou_path_file %s' %(sou_path_file))
                tar_path_file = sou_path_file.replace(source_folder,target_folder)
                tar_path = os.path.split(tar_path_file)[0]
                basename = os.path.basename(tar_path_file)
                # file_name = os.path.splitext(basename)[0]+ 'x4' + os.path.splitext(basename)[1]
                file_name = os.path.splitext(basename)[0]+ output_ext
                save_file = os.path.join(tar_path, file_name)
                # print(f'save_file:{save_file}')
                if not os.path.exists(tar_path):
                    os.makedirs (tar_path)
                imgBicubicScale(sou_path_file, save_file, scale)


source_folder = 'D:/AI_master/SR_datasets/vimeo_90k/vimeo_septuplet/sequences_B'
target_folder = 'D:/AI_master/SR_datasets/vimeo_90k/vimeo_septuplet/sequences_BIx4_B'
scale = 0.25
folder_images_resize(source_folder, target_folder, scale)