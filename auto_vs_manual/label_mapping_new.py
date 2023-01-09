from functools import reduce
from collections import Counter
import re

general_exclude = ['ctv','ptv','gtv', 'itv', 'prv', 'brachy']
include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']

def check_invalid_numeric_label(s):
    '''
    Labels like bladder_33 need to be removed, because they are usually interpolations. 100 is always an actual image, so keep those
    '''
    match = re.search(r'\d+$', s)
    if match:
        if match.group() != '100':
            return True
    return False

def is_bladder(text):
    text = text.lower()
    result = False
    if "bladder" in text or "blaas" in  text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['bt', 'bowel', 'mm', 'cm']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_hip(text): 
    text = text.lower()
    if 'hip' in text and not 'hippoc' in text and not 'prothese' in text:
        return True
    else:
        return False

def is_rectum(text):
    text = text.lower()
    result = False
    if "rectum" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "meso" in text:
        result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_bowel_bag(text):
    text = text.lower()
    result = False
    if "bowel_bag" in text or "bowel bag" in  text \
		or "bowel_loop" in text \
		or "bowel loop" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['cm', 'gy', 'cyste', 'legeblaas']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_sigmoid(text):
    text = text.lower()
    result = False
    if "sigmoid" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    for word in ['bt', 'mm', 'cm']:
        if word in text:
            result = False
    if check_invalid_numeric_label(text):
        result = False
    return result

def is_spinal_cord(text):
    text = text.lower()
    result = False
    if "spinal" in text:
        result = True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    return result

def is_anal_canal(text):
    text = text.lower()
    if 'anal' in text and "canal" in text:
        return True
    else:
        return False

def is_rectum_merged(text):
    """
    merges rectum and anal canal
    """
    text = text.lower()
    result = False
    if ('anal' in text and "canal" in text) or "rectum" in text:
        return True
    for exclude_word in general_exclude:
        if exclude_word in text:
            result = False
    if "bt" in text or "meso" in text:
        result = False
    if check_invalid_numeric_label(text):
        result = False
    return result


# ## Create and store label class mapping
# label_classes_flat = reduce(lambda x,y: x+y, label_classes)

# class_detectors = [
#     ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum), 
#     ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
#     ('anal_canal', is_anal_canal), ('bowel_bag', is_bowel_bag)
# ]
# merging rectum and anal_canal and calling both rectum
# class_detectors = [
#     ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum_merged), 
#     ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
#     ('bowel_bag', is_bowel_bag)
# ]

class_detectors = [
    ('bladder', is_bladder), ('hip', is_hip), ('rectum', is_rectum), ('anal_canal', is_anal_canal),
    ('spinal_cord', is_spinal_cord), ('sigmoid', is_sigmoid),
    ('bowel_bag', is_bowel_bag)
]


# mapping = {}

def map_label(label):    
    for class_name, class_detector in class_detectors:
        if class_detector(label):
            return class_name
    return None

#     mapping[class_name] = list(np.unique([label for label in label_classes_flat if class_detector(label)]))
