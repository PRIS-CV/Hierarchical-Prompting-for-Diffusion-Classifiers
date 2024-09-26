import pandas as pd
from diffusion.datasets import get_target_dataset

import csv

filename = 'prompts/Birds_name.csv'  # 替换为您的文件名

fine_grained_classes = {}  # 存储提取的fine-grained classes

with open(filename, 'r') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        idx = int(row['idx'])
        fine_grained_class = row['order'] #.strip("'")  # 去除首尾的单引号
        
        if 1 <= idx <= 13:
            fine_grained_classes[idx] = fine_grained_class

# 按照idx从1到196的顺序提取fine-grained classes
classes = [fine_grained_classes.get(i, '') for i in range(1, 14)]

templates = [
    'a photo of a {}, a type of bird.',
]
if __name__ == '__main__':
    dataset = 'cub'
    # target_dataset = get_target_dataset(dataset)

    prompt = [templates[0].format(cls) for cls in classes]
    classname = classes
    classidx = [i for i in range(13)]
    # sanity checks
    assert len(prompt) == len(classname) == len(classidx)
    # for i in range(len(prompt)):
    #     assert classname[i].lower().replace('_', '/') in prompt[
    #         i].lower(), f"{classname[i]} not found in {prompt[i].lower()}"

    # make pandas dataframe
    df = pd.DataFrame(data=dict(prompt=prompt,
                                classname=classname,
                                classidx=classidx))
    # save to csv
    df.to_csv(f'prompts/{dataset}_order_prompts.csv', index=False)
