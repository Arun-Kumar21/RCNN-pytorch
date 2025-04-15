import xml.etree.ElementTree as ET

# Function to parse VOC annotation
def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue

        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return objects, (width, height)