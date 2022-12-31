import PIL
from PIL import Image
import numpy as np
import json
import random
import copy

def imageretreival(path):
    inputfile = open(path,'r')
    jsondecode = json.load(inputfile)
    train_images = []
    for i in range(len(jsondecode["x"])):
        name = jsondecode["x"][i]
        #image_path = 'C:\\Users\\vaish\\Downloads\\capstone\\data\\celeba\\data\\raw\\img_align_celeba\\%s' %name
        image_path = 'C:\\Users\\amart\\OneDrive\\Desktop\\img_align_celeba\\%s' %name

        train_images.append(image_path)
    img_array = np.zeros((len(train_images),28,28), dtype = np.uint8)
    for i in range(len(train_images)):
        image = PIL.Image.open(train_images[i])
        image = image.convert('L')
        image = image.resize((28,28), Image.BICUBIC)
        img_array[i,:,:]= np.asarray(image)
    train_labels = jsondecode["y"]
    train_labels = np.array(train_labels, dtype='uint8')
    m1 = int(len(train_images)*0.7)
    #spilit to test,train
    train_l = train_labels[:m1]
    test_labels = train_labels[m1:]
    img_array = img_array.reshape([len(train_images),28,28,1])
    train_images = img_array.astype('float64')
    train_img = train_images[:m1]
    test_images = train_images[m1:]

    return(train_img,train_l,test_images,test_labels)

def randommask_alexnet(arr, a, b):
    l=0
    position = []
    maskedval = []
    for j in range(5):
        for k in range(5):
            for m in range(32):
                x = random.choice([0,1,1,a,b])
                if x ==1:
                    maskedval.append(arr[0][j][k][l][m])
                    position.append([0,j,k,l,m])
                
    for j in range(32):
        x = random.choice([0,1,1,a,b])
        if x ==1:
            maskedval.append(arr[1][j])
            position.append([1,j])
        
    for j in range(5):
        for k in range(5):
            for l in range(32):
                for m in range(64):
                    x = random.choice([0,1,1,a,b])
                    if x ==1:
                        maskedval.append(arr[2][j][k][l][m])
                        position.append([2,j,k,l,m])
                    
    for j in range(64):
        x = random.choice([0,1,1,a,b])
        if x ==1:
            maskedval.append(arr[3][j])
            position.append([3,j])
    
    for j in range(3136):
        for k in range(1024):
            x = random.choice([0,1,1,a,b])
            if x ==1:
                maskedval.append(arr[4][j][k])
                position.append([4,j,k])
                
    for j in range(1024):
        x = random.choice([0,1,1,a,b])
        if x ==1:
            maskedval.append(arr[5][j])
            position.append([5,j])
    
    for j in range(1024):
        for k in range(2):
            x = random.choice([0,1,1,a,b])
            if x ==1:
                maskedval.append(arr[6][j][k])
                position.append([6,j,k]) 
            
    for j in range(2):
        maskedval.append(arr[7][j])
        position.append([7,j])
        
    maskedval = np.array(maskedval)
    #position = np.array(position)
    return (maskedval, position)

def randommask(l,a,b):
    values=[]
    positions=[]
    curr_pos = []
    def rm_recursive(l,a,b,curr_pos):
        for i in range(len(l)):
            if isinstance(l[i],np.ndarray):
                curr_pos.append(i)
                rm_recursive(l[i],a,b,curr_pos)
            if isinstance(l[i],np.float32):
                #x = random.choice([0,1,1,a,b])
                if l[i] != 0 and random.choice([0,1,1,a,b])==1:
                    values.append(l[i])
                    temp=copy.deepcopy(curr_pos)
                    temp.append(i)
                    positions.append(temp)
                    continue
                l[i] *= 0
                #values.append(l[i])
                
                #temp=copy.deepcopy(curr_pos)
                #temp.append(i)
                #position.append(temp)

        if curr_pos != []:
            curr_pos.pop()
        return l

    rm_recursive(l,a,b,curr_pos)
    return values,positions