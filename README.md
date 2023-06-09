# IML_hack
IML 2023 Hackathon
files:

main.py - this file contains the main function. it divided into 3 parts.
part 1 - we upload the two models from the files. if we want to train the model we can 
make the files that save the models
part 2 - we open the tests data and preprocess it and predict the missing values
part 3 - we drew graphs that aggregating the data

task_1.py - in these file we have the preprocess of the data for the first task.
we also have a class that contains the model we want to save after it fit, and usefully data from the train preprocess.
also, we have prepare_train_1 function that called once when we want to fit the model.
also, we have execute_task_1 function that preproces the test and predict the missing value

task_2.py - very similar in purpose to task_1.py.
we have different preprocess and different model because here we want to do regression and not classification.

explore_data.py - some functions that do investigatory data analysis by creating plots selected attributes
from the dataset

agoda_cancellation_prediction.csv - this is the prediction of the cancellation. it is binary, 
1 for cancellation and 0 otherwise

agoda_cost_of_cancellation.csv - this is the predicted cost for all the cancellation.

model_1.joblib - the fitted model for the first task. this is classification model.

model_2.joblib - the fitted model for the second task. this is regression model.
