import fiftyone as fo
import fiftyone.zoo as foz

# Blog: https://blog.csdn.net/fengbingchun/article/details/121284157

# reference: https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html
datasets = foz.list_zoo_datasets()
print("available datasets:", datasets)

dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_name="evaluate-detections-tutorial")
dataset.persistent = True
session = fo.launch_app(dataset)

# print some information about the dataset
print("dataset info:", dataset)

# print a ground truth detection
sample = dataset.first()
print("ground truth:", sample.ground_truth.detections[0])

session.wait()