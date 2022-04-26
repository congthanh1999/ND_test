try:
    from src.system import face_recognition_system
except:
    from system import face_recognition_system

if __name__ == '__main__':
    dataset_path = 'D:/A jerry/Github/ND_test/datasets/'
    image_folder = 'D:/A jerry/Github/ND_test/datasets/images/'

    my_system = face_recognition_system(dataset_path, image_folder)

    my_system.index_dataset()
