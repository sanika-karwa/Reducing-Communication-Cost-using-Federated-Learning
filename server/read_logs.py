import pickle
import os

LOG_FILE_NAME = 'logs.dat'

if os.path.isfile(LOG_FILE_NAME):
    
    with open(LOG_FILE_NAME, "rb") as f:
        logs=pickle.load(f)
    
    '''for i in range(len(logs)):
        print(logs[i])
    '''

    print(logs)

else:
    print("File Doesn't Exist")
        