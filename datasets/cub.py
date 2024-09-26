import os
import os.path as op
import random
import typing as t

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase


template = ['a photo of a {}.']

class CUB_200_2011(DatasetBase):

    category_file: str = "CUB_200_2011_split/classes.txt"
    annotation_file: str = "CUB_200_2011_split/image_class_labels.txt"
    image_dir: str = "CUB_200_2011_split/images/"
    split_file: str = "CUB_200_2011_split/train_test_split.txt"
    images_list_file: str = "CUB_200_2011_split/images.txt" 

    def __init__(self, root, num_shots):
        self.template = template
        self.root = root
        assert op.exists(op.join(self.root, "CUB_200_2011")), "Please download the dataset by setting download=True."
        self.category2index, self.index2category = self._load_categories()
        self.annotations = self._load_annotations()
        train = self._load_samples("train")
        val = self._load_samples("test")
        test = self._load_samples("test")
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        subsample = "all" #cfg['subsample_classes']
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)

    def _load_annotations(self) -> dict:
        annotations = {}
        with open(op.join(self.root, self.annotation_file)) as f:
            lines = f.readlines()
        for line in lines:
            image_id, label = line.split()
            annotations[image_id] = int(label) - 1 
        return annotations

    def _load_samples(self, split) -> t.Union[t.List[str], dict]:
        image_ids = []
        samples = []
        items = []
        mode = '1' if split == "train" else '0'
        with open(op.join(self.root, self.split_file)) as f:
            lines = f.readlines()
        for line in lines:
            image_id, is_train = line.split()
            if mode == is_train:
                image_ids.append(image_id)

        with open(op.join(self.root, self.images_list_file)) as f:
            lines = f.readlines()
        
        category2sample = {v: [] for v in self.category2index.values()}

        for line in lines:
            image_id, image_path = line.split()
            if image_id in image_ids:
                image_path = op.join(self.root, self.image_dir, image_path)
                label = self.annotations[image_id]
                sample = (image_path, label)
                samples.append(sample)
                category2sample[int(label)].append(image_path)
                item = Datum(
                    impath=image_path,
                    label=label,
                    classname=self.index2category[int(label)]
                )
                items.append(item)
        return items
        

    def _load_categories(self) -> t.Union[dict, list]:
        category2index = dict()
        index2category = list()
        with open(op.join(self.root, self.category_file)) as f:
            lines = f.readlines()
        for line in lines:
            index, category = line.split()
            category2index[category] = int(index) - 1
            index2category.append(category)
        
        return category2index, index2category
