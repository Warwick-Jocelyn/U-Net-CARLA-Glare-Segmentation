import os

# 指定文件夹路径
folder_path = '/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Boda/Bayer_Glare/Glare_example/CameraRGB'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件是否包含要删除的字符串
#     if 'colorful_INSGT-' in filename:
#         # 创建新的文件名
#         new_filename = filename.replace('colorful_INSGT-', '')
#         # 获取完整的文件路径
#         old_file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_filename)
#         # 重命名文件
#         os.rename(old_file_path, new_file_path)
#         print(f'Renamed: {filename} -> {new_filename}')

from PIL import Image

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件是否是JPEG图像
#     if filename.lower().endswith('.jpg'):
#         # 构造完整的文件路径
#         jpg_file_path = os.path.join(folder_path, filename)
#         # 打开JPEG图像
#         with Image.open(jpg_file_path) as img:
#             # 构造新的PNG文件名
#             png_filename = filename.rsplit('.', 1)[0] + '.png'
#             png_file_path = os.path.join(folder_path, png_filename)
#             # 将图像转换为PNG并保存
#             img.save(png_file_path, 'PNG')
#             print(f'Converted: {filename} -> {png_filename}')

 
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是JPEG图像
    if filename.lower().endswith('.jpg'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 删除文件
        os.remove(file_path)
        print(f'Deleted: {filename}')
