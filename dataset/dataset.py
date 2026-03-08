import os

print(os.listdir(r"P:/Project/RPP_AI/dataset"))
def get_dataset_path():

    training_path = r"P:/Project/RPP_AI/dataset/Training"
    testing_path = r"P:/Project/RPP_AI/dataset/Testing"

    print("Training Path:", training_path)
    print("Testing Path:", testing_path)

    if not os.path.exists(training_path):
        raise Exception("Training folder not found")

    if not os.path.exists(testing_path):
        raise Exception("Testing folder not found")

    return training_path, testing_path