import os 
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

ROOT_DIR       = r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net'
DATA_ROOT_DIR  = os.path.join(ROOT_DIR, "Data")

TRAIN_DIR      = os.path.join(DATA_ROOT_DIR, 'train')
TEST_DIR       = os.path.join(DATA_ROOT_DIR, 'test')
VAL_DIR        = os.path.join(DATA_ROOT_DIR, 'validation')

TRAIN_IMAGES   = os.path.join(TRAIN_DIR, 'images') 
TRAIN_MASKED   = os.path.join(TRAIN_DIR, 'masked')

VAL_IMAGES     = os.path.join(VAL_DIR, 'images')
VAL_MASKED     = os.path.join(VAL_DIR, 'masked')

TEST_IMAGES    = os.path.join(TEST_DIR, 'images')
TEST_MASKED    = os.path.join(TEST_DIR, 'masked')

try:
    os.mkdir(DATA_ROOT_DIR)

    os.mkdir(TRAIN_DIR)
    os.mkdir(VAL_DIR)
    os.mkdir(TEST_DIR)

    os.mkdir(TRAIN_IMAGES)
    os.mkdir(TRAIN_MASKED)

    os.mkdir(VAL_IMAGES)
    os.mkdir(VAL_MASKED)

    os.mkdir(TEST_IMAGES)
    os.mkdir(TEST_MASKED)

except: 
    pass

def make_one_mask(original_path, id_name, new_path):
    image_path  = os.path.join(original_path, id_name, "images", id_name) + ".png"
    masked_path = os.path.join(original_path, id_name, "masks/")
    all_masks   = os.listdir(masked_path)

    image = cv2.imread(image_path, 1)   # Reading Image in RGB format

    mask = np.zeros((image.shape[0], image.shape[1], 1))

    for mask_filename in all_masks:
        _mask_path  = masked_path + mask_filename
        _mask_image = cv2.imread(_mask_path, -1)    # Reading image in grayscale format
        _mask_image = cv2.resize(_mask_image, (image.shape[1], image.shape[0]))   # Image_Size = 128 X 128
        _mask_image = np.expand_dims(_mask_image, axis = -1)    # Image_Dimensions = 128 X 128 X 1
        mask        = np.maximum(mask, _mask_image)
    
    masked_filename = os.path.join(new_path, "masked", id_name) + ".png"
    image_filename  = os.path.join(new_path, "images", id_name) + ".png"

    cv2.imwrite(masked_filename, mask)
    cv2.imwrite(image_filename, image)

def sort(ids, path, new_path, dataset='train'):
    print("Number of Images : " + str(len(ids)))
    
    if dataset == 'train':
        for id in ids:
            make_one_mask(path, id, new_path)
    
    if dataset == 'test':
        for id in ids:
            image_path  = os.path.join(path, id, "images",id) + ".png"
            image = cv2.imread(image_path, 1)
            image_filename  = os.path.join(new_path, "images", id) + ".png"
            cv2.imwrite(image_filename, image)

    

id      = os.listdir(r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Dataset\Train')
test_id = os.listdir(r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Dataset\Test')

train, val = train_test_split(id, test_size=0.1,train_size=0.9)

print(" ")

print("SORTING STARTED FOR TRAINING IMAGES..........")
sort(train, r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Dataset\Train', r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\train')
print("SORTING DONE")
print("NUMBER OF IMAGES SORTED : " + str(len(train)))

print(" ")

print("SORTING STARTED FOR VALIDATION IMAGES..........")
sort(val, r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Dataset\Train', r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\validation')
print("SORTING DONE")
print("NUMBER OF IMAGES SORTED : " + str(len(val)))

print(" ")

# for sorting testing data

print("SORTING STARTED FOR TESTING IMAGES..........")
sort(test_id, r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Dataset\Test', r'C:\Users\ragha\Desktop\Projects\Sematic Segmentation\U-Net\Data\test', dataset='test')
print("SORTING DONE")
print("NUMBER OF IMAGES SORTED : " + str(len(test_id)))

print(" ")





    






