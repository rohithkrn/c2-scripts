## create lmdb data-set

import numpy as np
from StringIO import StringIO
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

import pandas as pd
import skimage.io
import skimage.transform
import lmdb
import pdb

data_root = '/home/rohith/work/test_nets/distracted_driver_data/'

labels = pd.read_csv(data_root+'driver_imgs_list.csv').drop('subject', 1)[['img', 'classname']]
labels['img'] = labels.apply(lambda row: data_root+'imgs/train/'+row.classname+'/'+row.img, 1)
labels['classname'] = labels['classname'].map(lambda l: l[1])
labels = labels.reindex(np.random.permutation(labels.index))

labels.iloc[0:500].to_csv(data_root+'valid.txt', sep=' ', header=False, index=False)
labels.iloc[500:].to_csv(data_root+'train.txt', sep=' ', header=False, index=False)

train_input_file = data_root + 'train.txt'
train_db = data_root + 'data_train.minidb'
test_input_file = data_root + 'valid.txt'
test_db = data_root + 'data_test.minidb'

def write_db(db_type, db_name, input_file):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    with open(input_file,'r') as f:
    	data_paths = f.readlines()
    for j in range(len(data_paths)):
    	img_path = data_paths[j].split(' ')[0]
    	print(j,img_path)
        label = np.array(int(data_paths[j].split(' ')[1][0]))
        img = skimage.img_as_float(skimage.io.imread(img_path))
        img = skimage.transform.resize(img,(224,224))
        img = img[:,:,(2,1,0)]
        img_data = img.transpose(2,0,1)
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([utils.NumpyArrayToCaffe2Tensor(img_data), utils.NumpyArrayToCaffe2Tensor(label)])
        transaction.put('train_%04d'.format(j),feature_and_label.SerializeToString())
    del transaction
    del db

def create_db(output_file, input_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
    
    checksum = 0
    with env.begin(write=True) as txn:
        with open(input_file,'r') as f:
            valid_data = f.readlines()
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.HIP
        device_option.hip_gpu_id = 1
        with core.DeviceScope(device_option):
	        for j in range(len(valid_data)):
	            # MODIFY: add your own data reader / creator
	            img_path = valid_data[j].split(' ')[0]
	            label = int(valid_data[j].split(' ')[1][0])
	            print(j,img_path)
	            img = skimage.img_as_float(skimage.io.imread(img_path))
	            img = img.transpose(2,0,1)
	            img_data = skimage.transform.resize(img,(227,227))
	            

	            # Create TensorProtos
	            tensor_protos = caffe2_pb2.TensorProtos()
	            img_tensor = tensor_protos.protos.add()
	            #pdb.set_trace()
	            img_tensor.dims.extend(img_data.shape)
	            img_tensor.data_type = 1

	            flatten_img = img_data.reshape(np.prod(img_data.shape))
	            #pdb.set_trace()
	            img_tensor.float_data.extend(flatten_img)
	            #pdb.set_trace()
	            label_tensor = tensor_protos.protos.add()
	            label_tensor.data_type = 2
	            label_tensor.int32_data.append(label)
	            txn.put(
	                '{}'.format(j).encode('ascii'),
	                tensor_protos.SerializeToString()
	            )

            #checksum += np.sum(img_data) * label
                #if (j % 16 == 0):
                    #print("Inserted {} rows".format(j))

    #print("Checksum/write: {}".format(int(checksum)))
    #return checksum

if __name__ == "__main__":
    write_db("minidb",train_db,train_input_file)
    write_db("minidb",test_db,test_input_file)
    #create_db(train_lmdb,train_input_file)

