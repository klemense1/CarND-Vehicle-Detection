
import os
import glob

def train_test_split_procedual(listin):
    length = len(listin)
    idx_train_end = int(length*0.7)
    # print('idx_train_end', idx_train_end)
    idx_val_end = int(length*0.9)
    # print('idx_val_end', idx_val_end)

    training_data = listin[:idx_train_end]
    validation_data = listin[idx_train_end:idx_val_end]
    test_data = listin[idx_val_end:]
    return training_data, validation_data, test_data

def read_in_seperately(path_cars, path_notcars):
    # cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    # notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]

    # Loading + training,validation test split
    cars1 = glob.glob(path_cars + '/GTI_Far/*.png')
    cars2 = glob.glob(path_cars + '/GTI_MiddleClose/*.png')
    cars3 = glob.glob(path_cars + '/GTI_Left/*.png')
    cars4 = glob.glob(path_cars + '/GTI_Right/*.png')
    cars5 = glob.glob(path_cars + '/KITTI_extracted/*.png')

    notcars1 = glob.glob(path_notcars + '/Extras/*.png')
    notcars2 = glob.glob(path_notcars + '/GTI/*.png')

    train_data_cars = []
    val_data_cars = []
    test_data_cars = []

    train_data_no = []
    val_data_no = []
    test_data_no = []

    pix_dir_list_cars = [cars1, cars2, cars3, cars4, cars5]
    for pix_dir_cars in pix_dir_list_cars:
        tra_c, val_c, test_c = train_test_split_procedual(pix_dir_cars)
        train_data_cars = train_data_cars + tra_c
        val_data_cars = val_data_cars + val_c
        test_data_cars = test_data_cars + test_c

    pix_dir_list_no = [notcars1, notcars2]
    for pix_dir_no in pix_dir_list_no:
        tra_n, val_n, test_n = train_test_split_procedual(pix_dir_no)
        train_data_no = train_data_no + tra_n
        val_data_no = val_data_no + val_n
        test_data_no = test_data_no + test_n

    return train_data_cars, val_data_cars, test_data_cars, train_data_no, val_data_no, test_data_no

def read_in_together(path_cars, path_notcars):
    cars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_cars) for f in files if f.endswith('.png')]
    notcars = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(path_notcars) for f in files if f.endswith('.png')]

    return cars, notcars


if __name__ == "__main__":

    path_cars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/vehicles'
    path_notcars = '/Users/Klemens/Udacity_Nano_Car/P5_labeled_data/non-vehicles'
    train_data_cars, val_data_cars, test_data_cars, train_data_no, val_data_no, test_data_no = read_in_seperately(path_cars, path_notcars)

    #print(train_data_cars)

    cars, notcars = read_in_together(path_cars, path_notcars)
    #print(cars[0:8])
