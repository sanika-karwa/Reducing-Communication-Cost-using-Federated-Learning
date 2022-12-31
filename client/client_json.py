import socket
import os
import dataloader
from mlsocket import MLSocket
import threading


IP = socket.gethostbyname(socket.gethostname())
PORT = 2000
SERVER_ADDR = (IP, PORT)
FORMAT = 'utf-8'
SIZE = 1024
DATA_FILE = 'client_data'
NUM_CLIENTS = 30


def main():
    client=MLSocket()
    client.connect(SERVER_ADDR)

    #receive file name from server
    client_name = client.recv(SIZE).decode(FORMAT)
    current_path = os.getcwd()
    path = current_path + '//' + DATA_FILE + '//' + client_name

    #receive global model
    current_model = client.recv(SIZE)  

    #global_model = pickle.loads(global_model_pkl)


    #training model (return weigths in variable called 'new_model')
    #for the training model, we can send global_model, client_path(file location of client data)
    train_img, train_label, test_img, test_label = dataloader.imageretreival(path)
    num_datapoints = str(len(train_img)+len(test_img))
    #current_model = model_from_json(global_model)
    current_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    current_model.fit(train_img, train_label,batch_size=15,epochs=5,validation_data=(test_img, test_label))
    x = current_model.get_weights()
    new_weights, position =dataloader.randommask(x,0,0)
    #data = pickle.dumps(new_weights)
    #data1 = pickle.dumps(position)

    client.send(num_datapoints.encode(FORMAT))
    client.send(new_weights)
    client.send(position)



    '''
    #client_weights = current_model.get_weights()
    #id = client_name.split(".")[0]
    #current_model.save_weights("weights%s.h5" %id)
    current_model = current_model.to_json()
    with open("client_model.json","w") as json_file:
        json_file.write(current_model)

    #sending new weights 
    with open('client_model.json','rb') as f:
        current_model = json.load(f)
        client_weights_pkl = json.dumps(current_model)
    client.sendall(client_weights_pkl.encode(FORMAT))
    client.send(x.encode(FORMAT))
    #sendind the total number of samples in this client 
   '''
    
    print("Disconnected from the server...")
    client.close()
    
    
    
'''    
    while True:
        s_data = client.recv(SIZE).decode(FORMAT)
        print(f'{s_data}')
        print("main loop")
        c_data = input('> ')

        #recieving file from server
        if c_data.lower() == 'ready':
            client.send(c_data.encode(FORMAT))
            json_data = client.recv(SIZE)        
            json_data = pickle.loads(json_data)
            print(json_data)
            


        elif c_data.lower() == 'logout':
            client.send(c_data.encode(FORMAT))
            break
        
        
    print("Disconnected from the server.")
    client.close()
'''





'''if __name__ == "__main__":
    main()
'''

'''jobs = []
for i in range(NUM_CLIENTS):
    thread = threading.Thread(name=str(i),target=main())
    jobs.append(thread)
print(jobs)
'''


current_clients = 0
while(current_clients<=NUM_CLIENTS-1):
    current_clients+=1
    thread = threading.Thread(name=str(current_clients),target=main())
    thread.start()


