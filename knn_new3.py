from io import BytesIO
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pathlib import Path
import sys
import joblib
import logging
import face_recognition
sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[1]) + '/global_func')
from global_func.g_func import get_s3_object, get_filenames, push_object_to_s3,get_traceback

logger = logging.getLogger(__name__)

def train_s3_images_using_knn(s3_model_path, model_file_name,n_neighbors):
    X = []
    y = []
    targetsize = (256, 256)
    filenames, _ = get_filenames("employees/")
    print("Started Training")

    bugs = {}

    try:
        for filepath in filenames:
            if '/employee_images/' in filepath.lower():
                class_name = filepath.split('/')[1]
                s3_path = f"employees/{class_name}/employee_images/"
                file_name = filepath.split('/')[3]
                img, _ = get_s3_object(s3_path, file_name)
                image_content = img['Body'].read()
                image = Image.open(BytesIO(image_content)).resize(targetsize)

                # cropping the face from the image
                face_locations = face_recognition.face_locations(np.array(image))

                if not face_locations:
                    continue

                top, right, bottom, left = face_locations[0]
                face_image = np.array(image)[top:bottom, left:right]

                if face_image.size == 0:
                    continue

                img_array = face_image.flatten()

                X.append(img_array)
                y.append(class_name)

        # Ensure that each vector in X has the same length
        max_dimensions = max(vector.shape[0] for vector in X)
        X = [np.pad(vector, (0, max_dimensions - vector.shape[0]), mode='constant', constant_values=0)
             if vector.shape[0] < max_dimensions else vector for vector in X]

        knn_classifier = KNeighborsClassifier(n_neighbors)
        knn_classifier.fit(X, y)

        # Pushing the model to S3
        model_content = BytesIO()
        joblib.dump(knn_classifier, model_content)

        push_status, push_bugs = push_object_to_s3(model_content.getvalue(), s3_model_path, model_file_name)

        if push_status:
            print(f"Model successfully pushed to S3 at {s3_model_path}{model_file_name}.")
        else:
            print("Error while pushing the model to S3. Check logs for details.")
            print("Bugs:", push_bugs)
            bugs.update(push_bugs)

    except Exception as e:
        tb = get_traceback()  
        bugs[1] = {}
        bugs[1]['code'] = "00" 
        bugs[1]['message'] = f"Error during training: {e}. Traceback: {tb}"
        bugs[1]['input'] = {
            'n_neighbors': n_neighbors,
            's3_model_path': s3_model_path,
            'model_file_name': model_file_name
        }
        print(f"Error during training: {e}. Check logs for details.")
        print("Bugs:", bugs)
        logger.error("Error during training: %s. Traceback: %s", e, tb)

    return bugs

n_neighbors = #enter an integer ranging from 2 to 6 depending upon the dataset
s3_model_path = "employees/face_id_models/"
model_file_name = ""
bugs = train_s3_images_using_knn( s3_model_path, model_file_name,n_neighbors)
