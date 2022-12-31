# Reducing-Communication-Cost-using-Federated-Learning

A research project that tries to improve the effectiveness of the federated learning environment by lowering communication overhead for each client by randomly masking it and transmitting it from various clients to a central server that is connected using socket programming.

The dataset used is the CelebA image dataset to verify this. 

### Instructions to run

Clone the repository,
```sh
$ git clone https://github.com/sanika-karwa/Reducing-Communication-Cost-using-Federated-Learning.git
```

To Start Server -> Within Server Folder run command 
```sh
$ cd server
$ py server.py
```

To Start Client when Server Starts Listening during each FL Round -> Within Client Folder run command 
```sh
$ cd client
$py client.py
```

After Federated Learning Completion -> Within Server Folder run command 
```sh
$ cd server
$ py read_logs.py > logs.txt
```

Logs Data will be Generated into the .txt File
