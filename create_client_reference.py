import os
import pickle

TARGET_FILE = 'federated_data_details'

client_references = {}
num_clients = 0

#current path
path = os.getcwd()
#path of data folder
path = path + '\\' + TARGET_FILE

for (dirpath, dirnames, filenames) in os.walk(path):
    for name in filenames:
        client_references[name] = {"active" : 0, "time_taken": 0, "size_model": 0}
        num_clients += 1
    break
client_references["num_clients"] = num_clients
client_references['act_clients'] = 0
print(client_references)

#saving client references
with open('client_references.pickle', 'wb') as file:
    pickle.dump(client_references, file)



