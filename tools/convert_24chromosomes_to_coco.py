import xml.etree.ElementTree as ET
import os
import json
import random

data_root = 'data/24_chromosomes_object'
ann_dir = os.path.join(data_root, 'annotations')
img_dir = os.path.join(data_root, 'JEPG')
output_dir = os.path.join(data_root, 'coco')

CATEGORIES = [
    'A1', 'A2', 'A3', 'B4', 'B5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'D13', 'D14', 'D15', 'E16', 'E17', 'E18', 'F19', 'F20', 'G21', 'G22', 'X', 'Y',
]
CAT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CATEGORIES)}

random.seed(42)
all_xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
random.shuffle(all_xml_files)

n_total = len(all_xml_files)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.1)

train_files = all_xml_files[:n_train]
val_files = all_xml_files[n_train:n_train + n_val]
test_files = all_xml_files[n_train + n_val:]

print(f'Total: {n_total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')


def convert_voc_to_coco(xml_files, split_name):
    images = []
    annotations = []
    ann_id = 1

    for img_id, xml_file in enumerate(xml_files, start=1):
        xml_path = os.path.join(ann_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        images.append({
            'id': img_id,
            'file_name': filename,
            'width': width,
            'height': height,
        })

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CAT_NAME_TO_ID:
                print(f'Warning: unknown category {name} in {xml_file}, skipping')
                continue

            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': CAT_NAME_TO_ID[name],
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': difficult,
            })
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': i + 1, 'name': name, 'supercategory': 'chromosome'}
            for i, name in enumerate(CATEGORIES)
        ],
    }

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(coco, f, indent=2)

    for img_info in images:
        src = os.path.join(img_dir, img_info['file_name'])
        dst = os.path.join(split_dir, img_info['file_name'])
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

    print(f'{split_name}: {len(images)} images, {len(annotations)} annotations')
    return coco


convert_voc_to_coco(train_files, 'train')
convert_voc_to_coco(val_files, 'valid')
convert_voc_to_coco(test_files, 'test')
print('Done!')
