#%%
# fetch model from s3 and test
from PIL import Image
import face_recognition, joblib 
import numpy as np
from io import BytesIO
from pathlib import Path
import sys
import logging
from datetime import datetime
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[1]) + '/global_func')
from global_func.g_func import get_s3_object, get_traceback, combine_bugs, get_parquet_df, push_df_to_s3

def fetch_s3_model(s3_path, model_name):
    bugs={}
    try:
        # fetching the model from S3
        s3_model_object, bugs_local = get_s3_object(s3_path, model_name)
        bugs = combine_bugs(bugs, bugs_local)


        if s3_model_object is not None:
            # loading the trained KNN model from S3
            s3_model_content = s3_model_object['Body'].read()
            knn_classifier = joblib.load(BytesIO(s3_model_content))

            return knn_classifier
        else:
            print("Failed to fetch the model from S3.")
            return None

    except Exception as e:
        print(f"An error occurred while fetching the model: {e}")
        tb = get_traceback()
        bugs_local={}
        bugs_local[1]={}
        bugs_local[1]["message"]=f"An error occurred while fetching the model: {e} , {tb}"
        bugs_local[1]['input'] = s3_path
        bugs_local[1]['input'] = model_name
        bugs = combine_bugs(bugs, bugs_local)
        logging.error("Error fetching model from S3: %s. Traceback: %s ,s3_path: %s, model_name: %s", e, tb,s3_path,model_name)
        

def test_s3_model(knn_classifier,s3_path, test_image, model_name, actual_label, parquet_filename):
    bugs = {}

    try:
        existing_df, _ = get_parquet_df(s3_path, parquet_filename)
        if existing_df is None:
            existing_df = pd.DataFrame(columns=['image_name', 'model_name', 'actual_label', 'predicted_label', 'timestamp', 'is_correct', 'is_face_detected'])
        
        if knn_classifier is not None:
            target_dimensions = 49923
            
            test_image = test_image.convert("RGB")

            face_locations = face_recognition.face_locations(np.array(test_image))

            # checking if a face is detected
            is_face_detected = bool(face_locations)

            # setting a default value for predicted_label when no face is detected
            predicted_label = "NoFaceDetected" if not is_face_detected else None

            if not is_face_detected:
                
                print("No face found in the test image.")

                image_name = Path(test_image_path).name

                timestamp = datetime.now()
                
                new_row = {'image_name': image_name, 'model_name': model_name, 'actual_label': actual_label,
                           'predicted_label': predicted_label, 'timestamp': timestamp, 'is_correct': False,
                           'is_face_detected': is_face_detected}

                # appending
                updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)

                print(updated_df)

                status, bugs_local = push_df_to_s3(updated_df, s3_path, parquet_filename)
                bugs = combine_bugs(bugs, bugs_local)
                if not status:
                    print(f"Failed to push DataFrame to S3. Bugs: {bugs}")

                return

            top, right, bottom, left = face_locations[0]
            test_face = np.array(test_image)[top:bottom, left:right]

            test_image_array = test_face.flatten()

            if test_image_array.size < target_dimensions:
                
                test_image_array = np.pad(test_image_array, (0, target_dimensions - test_image_array.size), mode='constant', constant_values=0)
            elif test_image_array.size > target_dimensions:
                
                test_image_array = test_image_array[:target_dimensions]

            test_image_array = test_image_array.reshape(1, -1)

            predicted_label = knn_classifier.predict(test_image_array)[0]

            image_name = Path(test_image_path).name

            timestamp = datetime.now()

            # checking if the prediction is correct
            is_correct = actual_label == predicted_label

            # creating a new row for the DataFrame
            new_row = {'image_name': image_name, 'model_name': model_name, 'actual_label': actual_label,
                       'predicted_label': predicted_label, 'timestamp': timestamp, 'is_correct': is_correct,
                       'is_face_detected': is_face_detected}

            # appending
            updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)

            print(updated_df)

            print(f"The predicted label : {predicted_label}")

            status, bugs_local = push_df_to_s3(updated_df, s3_path, parquet_filename)
            bugs = combine_bugs(bugs, bugs_local)
            if not status:
                print(f"Failed to push DataFrame to S3. Bugs: {bugs}")

        else:
            print("Failed to fetch the model from S3.")

    except Exception as e:
        print(f"An error occurred: {e}")
        tb = get_traceback()
        bugs_local = {}
        bugs_local[1] = {}
        bugs_local[1]['message'] = f"Error during testing S3 model. Error: {e}. Traceback: {tb}"
        bugs_local[1]['input'] = {}
        bugs_local[1]['input']['actual_label'] = actual_label
        bugs_local[1]['input']['model_name'] = model_name
        bugs_local[1]['input']['s3_path'] = s3_path
        bugs = combine_bugs(bugs, bugs_local)
        logging.error("Error during testing S3 model: %s,s3_path: %s, Error: %s. Traceback: %s", actual_label,s3_path, e, tb)


s3_path = "employees/face_id_models/"
model_name = "00001.joblib"
test_image_path = r"C:\Users\arvin\OneDrive\Desktop\MobileNetV2\database\training\sairam\46.jpg"
actual_label = "00013"
parquet_filename = "predictions.parquet"

# Open and resize the image using PIL
target = (256, 256)
pil_image = Image.open(test_image_path).resize(target)

knn_classifier = fetch_s3_model(s3_path, model_name)

# Test the model
test_s3_model(knn_classifier,s3_path, pil_image, model_name, actual_label, parquet_filename)


#%%
from PIL import Image
import face_recognition, joblib 
import numpy as np
from io import BytesIO
from pathlib import Path
import sys
import logging
from datetime import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
test_image_path = r"C:\Users\arvin\OneDrive\Desktop\MobileNetV2\database\training\sairam\46.jpg"

# Open and resize the image using PIL
target = (256, 256)
pil_image = Image.open(test_image_path).resize(target)

# Display the resized image
pil_image.show()

# %%
