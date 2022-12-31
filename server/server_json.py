import socket
import threading
import pickle
import random
import time
import os
import pathlib
from keras.models import load_model
import dataloader
import sys
from mlsocket import MLSocket


IP = socket.gethostbyname(socket.gethostname())
PORT = 2000
SERVER_ADDR = (IP, PORT)
SIZE = 1024
FORMAT = 'utf-8'
CURRENT_PATH = pathlib.Path(os.getcwd())            #type -> path
PARENT_PATH = CURRENT_PATH.parent.absolute()        #type -> path
SERVER_DATA_FILE = 'server_data.json'
PATH = "C:\\Users\\vaish\\Downloads\\Socket_Prog\\client\\"
NUM_COMM_ROUNDS = 100
COMM_TIMEOUT_EACH_ROUND = 100       #has to be larger than processing time for the model
LOG_FILE_NAME = 'logs.dat'
MODEL_NAME = dataloader.create_vgg_model

model = MODEL_NAME()
weightss = model.get_weights()
struct =[abs(i*0) for i in weightss]
lock = threading.Lock()

def before_FED():
    #train_img, train_label, test_img, test_label = dataloader.imageretreival(str(CURRENT_PATH) + '//' + '17.json')
    train_img, train_label, test_img, test_label = dataloader.imageretreival(str(CURRENT_PATH) + '//' + 'server_data.json')
    num_datapoints = len(train_img)+len(test_img)
    first_model = MODEL_NAME()
    first_model.fit(train_img, train_label,batch_size=15,epochs=5,validation_data=(test_img, test_label))
    #first_model_json = first_model.to_json()
    #with open('global_model.json','w') as json_file:
    #    json_file.write(first_model_json)
    
    first_model.save("global_model.h5")
    first_model.save("vgg_model_pre_fl.h5")

    temp = first_model.get_weights()
    with open("weight",'wb') as x:
        pickle.dump(temp,x)
    with open("num_train_datapoints",'wb') as y:
        pickle.dump(num_datapoints,y)
    loss,acc=first_model.evaluate(train_img,train_label,verbose=2)
    print(f"loss:{loss},accuracy:{acc}")


def handle_client(conn, addr):
    print(f'{addr} connected...')
    current_thread_name = threading.currentThread().getName()
    #sending file name to client
    conn.send(current_thread_name.encode(FORMAT))
    
    #sending current global model
    #read global model
    #with open('global_model.json','rb') as f:
    #    current_model = json.load(f)
    #    current_model_pkl = json.dumps(current_model)
    
    current_model = load_model('global_model.h5')
    t1=time.perf_counter()
    #start timing
    #send weights to client
    #conn.send(current_model_pkl.encode('utf-8'))
    conn.send(current_model)

    

    #receiving new weights 
    num_datapoints = int(conn.recv(SIZE).decode(FORMAT))
    #receive new weights
    new_weights = conn.recv(SIZE)
    
    #new_weights_pkl = pickle.loads(new_weights)
    #receiving the positions of new weights
    position = conn.recv(SIZE)
    #position_pkl = pickle.loads(position)

    #regenerating client weights from list and positions 
    client_weights = dataloader.regen(struct, new_weights,position)
    #client_references[current_thread_name]['size_model'] = size 
    print("Data received")


    lock.acquire()

    #fedavg
    #weights here 
    with open("weight",'rb') as x:
        global_weights = pickle.load(x)
    with open("num_train_datapoints",'rb') as y:
        num_datapoints_global = pickle.load(y)

    #global_weights = dataloader.fedavg(global_weights,client_weights,int(train))
    global_weights,num_datapoints_global_new = dataloader.fedavg(global_weights,client_weights,num_datapoints,num_datapoints_global)

    #stop timing
    t2=time.perf_counter()

    
    #print(current_thread_name, global_weights[0][0][0][0])
    with open("weight",'wb') as x:
        pickle.dump(global_weights,x)
    with open("num_train_datapoints",'wb') as y:
        pickle.dump(num_datapoints_global_new,y)

    lock.release()

    #model size
    sum_size = 0
    for i in range(len(client_weights)):
        sum_size += sys.getsizeof(client_weights[i])
    model_size,temp = dataloader.format_bytes(sum_size)
    
    #save model
    #total communication time (sending weights, training model, receiving weights) in seconds
    elapsed_time_2 = (t2-t1)
    #time.append(elapsed_time)
    key = current_thread_name
    if(type(client_references[key])) == dict :
        client_references[key]['time_taken_including_processing'] = elapsed_time_2
        client_references[key]['model_size'] = model_size
    
    print(f'Total communication time = {elapsed_time_2}')

    #client_references[current_thread_name]['weights'] = new_weights

    print(f'{addr} disconnected...')
    conn.close()



def main():   
    
    try:
        os.remove(LOG_FILE_NAME) 
    except FileNotFoundError:
        pass

    print("Before Federated Learning...")
    before_FED()
    
    print("Starting Federated Learning...")
    for i in range(NUM_COMM_ROUNDS):
        print(f"--------- START OF COMMUNICATION ROUND {i+1}----------")
        try:
            print('Server is starting...')
            server=MLSocket()
            server.bind(SERVER_ADDR)
            server.listen()
            server.settimeout(COMM_TIMEOUT_EACH_ROUND)
            print(f'Server is listening on {IP}:{PORT}...')
            d = 0

            while (True):
                conn, addr = server.accept()
                
                keys = list(client_references.keys())
                while True:
                    current = random.choice(keys)
                    if(current != 'num_clients' and current != 'act_clients' and client_references[current]['active'] == 0):
                        client_references['act_clients'] +=1
                        client_references[current]['active']+=1
                        break

                thread = threading.Thread(name=current,target=handle_client, args=(conn, addr))
                thread.start()
                d+=1
                print("Number of clients = ",d)
                #print(f'Number of active connections - {threading.activeCount() - 1}')
        
        except socket.timeout:
            server.close()
        
        #.join()
        print(f"--------- END OF COMMUNICATION ROUND {i+1}----------")
        #at the end of a commuinication round 
        global_model = MODEL_NAME()
        with open("weight",'rb') as x:
            global_weights = pickle.load(x)
        global_model.set_weights(global_weights)
        #getting the accuracy of the global model
        #train_img, train_label, test_img, test_label = dataloader.imageretreival(str(CURRENT_PATH) + '//' + '20.json')
        train_img, train_label, test_img, test_label = dataloader.imageretreival(str(CURRENT_PATH) + '//' + 'server_data.json')
        scores = global_model.evaluate(train_img,train_label,verbose = 2)
        global_model.save("global_model.h5")

        #logging the time taken and active clients and size
        logd = {}
        for key in client_references:
            if(type(client_references[key])) == dict and client_references[key]['active'] != 0:
                logd[key] = client_references[key]
        logd['act_clients'] = client_references['act_clients']
        acc=scores[1]*100
        logd["accuracy"]=acc

        if os.path.isfile(LOG_FILE_NAME)==False:   #for first federated round
            logs = []

        else:
            with open(LOG_FILE_NAME, "rb") as f:
                logs=pickle.load(f)
        
        logs.append(logd)

        with open(LOG_FILE_NAME, "wb") as f:
            pickle.dump(logs, f)
        
        #resetiing the dictionary that logs
        for key in client_references:
            if(type(client_references[key])) == dict :
                client_references[key]['active'] = 0
        client_references['act_clients'] = 0




with open('client_references.pickle', 'rb') as file:
    client_references = pickle.load(file)


if __name__ == '__main__':
    main()
