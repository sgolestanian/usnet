import json
from os import makedirs, path, walk
from datetime import datetime
from glob import glob
import tensorflow as tf
import progressbar

DATASET_METADATA = {
    'name': 'usnet_dataset',
    'version': '1',
    'date_of_creation': datetime.now().strftime('%Y-%m-%d'),
    'train_data_count': 0,
    'validation_data_count': 0,
    'data_source': 'CAMUS',
    'discription': '2D Ultrasound data with segmentation masks',
    'path': 'data',
    'list_files':[]
}

VIEWS = ['2CH', '4CH']
TIMING = ['ED', 'ES']


def write_json(data, json_path):
    json_object = json.dumps(data, indent = 4)
    with open(json_path, "w") as outfile:
        outfile.write(json_object)



complete_dataset_name = DATASET_METADATA['name'] + '__' + DATASET_METADATA['version']
print(f"Creating dataset: {DATASET_METADATA['name']}")
print(f"Version         : {DATASET_METADATA['version']}")
print(f"DATE            : {DATASET_METADATA['date_of_creation']}")


dataset_path = path.join(DATASET_METADATA['path'], complete_dataset_name)

# makedirs(dataset_path)



train_patients = glob('data/datasets/training/*')
validation_patients = glob('data/datasets/validation/*')



def read_cfg(patient, view = '2CH'):
    """
    This function reads the patients' data

    Parameters
    -----
        patient: str or path-like
            Address to patient's folder

        view: str
            Which view to use. It should be 2CH or 4CH

    Returns
    -----
        patient_data: dict
            a dict with patient's data.
    """
    info_file_name = 'Info_'+view+'.cfg'
    info_file_path = path.join(patient, info_file_name)
    
    def remove_newline(line):
        return line[:-1]
    
    def keyvalue_pair(data):
        [key, value_str] = data.split(':')
        value_str = value_str.strip()
        return [key, value_str]
    
    with open(info_file_path, 'r') as file:
        patient_data = dict()
        info_string = file.readlines()
        info_string = list(map(remove_newline, info_string))
        info = list(map(keyvalue_pair, info_string))
        patient_data.update({'ed':float(info[0][1])})
        patient_data.update({'es':float(info[1][1])})
        patient_data.update({'nframe':int(info[2][1])})
        patient_data.update({'sex':info[3][1]})
        patient_data.update({'age':int(info[4][1])})
        patient_data.update({'image_quality':info[5][1]})
        patient_data.update({'lvedv':float(info[6][1])})
        patient_data.update({'lvesv':float(info[7][1])})
        patient_data.update({'lvef':float(info[8][1])})
        patient_data.update({'view':view})
        
        
    return patient_data


def tf_read_raw_image(image_path):
    imgfile = tf.io.read_file(image_path)
    img_fileadd = tf.strings.split(image_path, '.')[0]
    mhd_fileadd = tf.strings.join([img_fileadd, '.mhd'])
    img_dims = tf_read_mhd_data(mhd_fileadd)
    def correct_dims(dims):
        dims_correct = [dims[1], dims[0], dims[2]]
        dims_correct = tf.stack(dims_correct, axis=0)
        return dims_correct
    imgbytes = tf.io.decode_raw(imgfile, out_type=tf.uint8)
    img = tf.reshape(imgbytes, correct_dims(img_dims))
    return img.numpy()



def tf_read_mhd_data(mhd_address):
    mhd_data_file = tf.io.read_file(mhd_address)
    mhd_lines = tf.strings.split(mhd_data_file, '\n')
    mhd_lines = tf.strings.split(mhd_lines, ' = ')
    dimsize = mhd_lines[10][1]
    dimsize = tf.strings.split(dimsize, ' ')
    dimsize = tf.boolean_mask(dimsize, mask=tf.math.logical_not(tf.math.equal(dimsize, b'')))
    #dimsize_new = tf.map_fn(tf.strings.to_number, dimsize)
    dimsize = tf.strings.to_number(dimsize)
    dimsize = tf.cast(dimsize, tf.int32)
    #element_number_of_channels = tf.strings.to_number(mhd_lines[11][1])
    return dimsize.numpy()


def tf_read_mask_image(image_path):
    imgfile = tf.io.read_file(image_path)
    img_fileadd = tf.strings.split(image_path, '.')[0]
    mhd_fileadd = tf.strings.join([img_fileadd, '.mhd'])
    img_dims = tf_read_mhd_data(mhd_fileadd)
    def correct_dims(dims):
        dims_correct = [dims[1], dims[0], dims[2]]
        dims_correct = tf.stack(dims_correct, axis=0)
        return dims_correct
    imgbytes = tf.io.decode_raw(imgfile, out_type=tf.uint8)
    img = tf.reshape(imgbytes, correct_dims(img_dims))
    return img


def tf_write_image(path, x):
    img = tf.io.encode_png(x)
    tf.io.write_file(path, img)

subset = 'train'
for patient in train_patients:
    DATA_INSTANCE = {}
    DATA_INSTANCE['patient_id'] = path.basename(patient)
    DATA_INSTANCE['subset'] = subset
    for view in VIEWS:
        DATA_INSTANCE['patient_metadata'] = read_cfg(patient, view)
        DATA_INSTANCE['view'] = view
        for time in TIMING:
            data_name = f"{DATA_INSTANCE['patient_id']+'_'+view+'_'+time}"

            DATA_INSTANCE['TIMING'] = time
            DATA_INSTANCE['IMG_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'.raw'
            DATA_INSTANCE['IMGMHD_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'.mhd'
            DATA_INSTANCE['MASK_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'_gt.raw'
            DATA_INSTANCE['MASKMHD_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'_gt.mhd'
            DATA_INSTANCE['IMG_DIMS'] = tf_read_mhd_data(path.join(patient, DATA_INSTANCE['IMGMHD_FILE'])).tolist()

            image = tf_read_raw_image(path.join(patient, DATA_INSTANCE['IMG_FILE']))
            data_image_name = data_name+'.png'
            DATA_INSTANCE['IMG_FILE'] = data_image_name
            tf_write_image(path.join(dataset_path, data_image_name), image)

            mask = tf_read_mask_image(path.join(patient, DATA_INSTANCE['MASK_FILE']))
            data_mask_name = data_name+'_gt.png'
            DATA_INSTANCE['MASK_FILE'] = data_mask_name
            tf_write_image(path.join(dataset_path, data_mask_name), mask)

            DATASET_METADATA['train_data_count'] += 1
            
            DATASET_METADATA['list_files'].append(data_name+'.json')
            write_json(DATA_INSTANCE, path.join(dataset_path, data_name+'.json'))


subset = 'validation'
for patient in validation_patients:
    DATA_INSTANCE = {}
    DATA_INSTANCE['patient_id'] = path.basename(patient)
    DATA_INSTANCE['subset'] = subset
    for view in VIEWS:
        DATA_INSTANCE['patient_metadata'] = read_cfg(patient, view)
        DATA_INSTANCE['view'] = view
        for time in TIMING:
            data_name = f"{DATA_INSTANCE['patient_id']+'_'+view+'_'+time}"

            DATA_INSTANCE['TIMING'] = time
            DATA_INSTANCE['IMG_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'.raw'
            DATA_INSTANCE['IMGMHD_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'.mhd'
            DATA_INSTANCE['MASK_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'_gt.raw'
            DATA_INSTANCE['MASKMHD_FILE'] = DATA_INSTANCE['patient_id']+'_'+view+'_'+time+'_gt.mhd'
            DATA_INSTANCE['IMG_DIMS'] = tf_read_mhd_data(path.join(patient, DATA_INSTANCE['IMGMHD_FILE'])).tolist()

            image = tf_read_raw_image(path.join(patient, DATA_INSTANCE['IMG_FILE']))
            data_image_name = data_name+'.png'
            DATA_INSTANCE['IMG_FILE'] = data_image_name
            tf_write_image(path.join(dataset_path, data_image_name), image)

            mask = tf_read_mask_image(path.join(patient, DATA_INSTANCE['MASK_FILE']))
            data_mask_name = data_name+'_gt.png'
            DATA_INSTANCE['MASK_FILE'] = data_mask_name
            tf_write_image(path.join(dataset_path, data_mask_name), mask)

            DATASET_METADATA['validation_data_count'] += 1
            
            DATASET_METADATA['list_files'].append(data_name+'.json')
            write_json(DATA_INSTANCE, path.join(dataset_path, data_name+'.json'))



write_json(DATASET_METADATA, path.join(dataset_path, 'info.json'))
