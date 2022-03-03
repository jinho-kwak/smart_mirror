import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import traceback

from xml.etree.ElementTree import Element, SubElement, ElementTree, dump

class LogGetter:
    log = ''

    def center(self, s, n=100):
        self.log += s.center(n)


def align_center(log_str, max_len=100):
    log_str = log_str.split('\n')
    return '\n'.join([row.center(max_len, ' ') for row in log_str])
# def save_img_xml_s3_or_local(save_path:str, xml_coor:list, s3, is_save:bool, s3_Bucket_name:str, shelf_storage_dict, re_img):
#     split_list = save_path.split('/')
#     companyId = split_list[2]
#     storeId = split_list[3]
#     deviceId = split_list[4]
#     floor = split_list[5]
#     camera = split_list[6]
#     img = re_img.get(f'{companyId}_{storeId}_{deviceId}_f{floor}_cam{camera}')
#     encoded_img = np.frombuffer(img, dtype=np.uint8) 
#     image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     if shelf_storage_dict == 'EC':
#         image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전
#     for idx in xml_coor:
#         cv2.rectangle(image, (idx[0],idx[1]), (idx[2],idx[3]), (0,0,255), 3)
#     if is_save == True:
#         s3.put_object(Body=cv2.imencode('.jpg', image)[1].tostring(), Bucket=s3_Bucket_name, Key=save_path)
#     else:
#         cv2.imwrite(f'{save_path}', image)
    
def save_img_xml_s3_or_local(save_path:str, xml_coor:list, s3, is_save:bool, s3_Bucket_name:str, shelf_storage_dict, image):
    split_list = save_path.split('/')
    encoded_img = np.frombuffer(image[1], dtype=np.uint8) 
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for idx in xml_coor:
        cv2.rectangle(image, (idx[0],idx[1]), (idx[2],idx[3]), (0,0,255), 3)
    if is_save == True:
        s3.put_object(Body=cv2.imencode('.jpg', image)[1].tostring(), Bucket=s3_Bucket_name, Key=save_path)
    else:
        cv2.imwrite(f'{save_path}', image)



def save_xml(saved_path:str, trDate:str, xml_coor:list, s3, is_save:bool, s3_Bucket_name:str):
    createFolder(saved_path)
    root = Element('annotation')
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = '1920'
    SubElement(size, 'height').text = '1080'
    SubElement(size, 'depth').text = '3'
    SubElement(root, 'segmented').text = '0'
    for i in xml_coor:
        obj = SubElement(root, 'object')
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(i[0])
        SubElement(bbox, 'ymin').text = str(i[1])
        SubElement(bbox, 'xmax').text = str(i[2])
        SubElement(bbox, 'ymax').text = str(i[3])
    xml_str = ET.tostring(root, encoding='utf8', method='xml')
    if is_save == True:
        s3.put_object(Body=bytes(xml_str), Bucket=s3_Bucket_name, Key=f'{saved_path}/{trDate}.xml', ContentType = 'text/xml')
    else:
        ElementTree(root).write(f'{saved_path}/{trDate}.xml')
        
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        traceback.print_exc()