import os
import cv2
import json
import xml.etree.ElementTree as ET

# Paths
img_dir = "datasets/VOC2012/JPEGImages"
ann_dir = "datasets/VOC2012/Annotations"
out_img_dir = "VOC2012-Visualized"
out_json_dir = "VOC2012-JSON"

# Create output directories if not exist
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_json_dir, exist_ok=True)

# Function to parse Pascal VOC annotation
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append({
            "class": name,
            "bbox": [xmin, ymin, xmax, ymax]
        })
    
    return objects

# Loop through all annotation files
for ann_file in os.listdir(ann_dir):
    if not ann_file.endswith(".xml"):
        continue
    
    # Image file name (same base name but .jpg)
    img_id = os.path.splitext(ann_file)[0]
    img_path = os.path.join(img_dir, img_id + ".jpg")
    ann_path = os.path.join(ann_dir, ann_file)
    
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found, skipping.")
        continue
    
    # Load image
    img = cv2.imread(img_path)
    
    # Parse annotation
    objects = parse_voc_xml(ann_path)
    
    # Draw bounding boxes
    for obj in objects:
        label = obj["class"]
        xmin, ymin, xmax, ymax = obj["bbox"]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save visualized image
    out_img_path = os.path.join(out_img_dir, img_id + ".jpg")
    cv2.imwrite(out_img_path, img)
    
    # Save JSON with bbox values
    json_data = {
        "image": img_id + ".jpg",
        "objects": objects
    }
    out_json_path = os.path.join(out_json_dir, img_id + ".json")
    with open(out_json_path, "w") as jf:
        json.dump(json_data, jf, indent=4)

print("✅ Done! Images saved in", out_img_dir, "and JSON files in", out_json_dir)
