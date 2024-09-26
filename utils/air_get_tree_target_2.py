import numpy as np
import torch
from torch.autograd import Variable


trees = [
[1, 1, 1],
[2, 2, 1],
[3, 3, 1],
[4, 3, 1],
[5, 3, 1],
[6, 3, 1],
[7, 4, 1],
[8, 4, 1],
[9, 5, 1],
[10, 5, 1],
[11, 5, 1],
[12, 5, 1],
[13, 6, 1],
[14, 7, 2],
[15, 8, 3],
[16, 9, 3],
[17, 10, 7],
[18, 10, 7],
[19, 11, 7],
[20, 12, 4],
[21, 13, 5],
[22, 14, 5],
[23, 15, 5],
[24, 16, 5],
[25, 16, 5],
[26, 16, 5],
[27, 16, 5],
[28, 16, 5],
[29, 16, 5],
[30, 16, 5],
[31, 16, 5],
[32, 17, 5],
[33, 17, 5],
[34, 17, 5],
[35, 17, 5],
[36, 18, 5],
[37, 18, 5],
[38, 19, 5],
[39, 19, 5],
[40, 19, 5],
[41, 20, 5],
[42, 20, 5],
[43, 21, 21],
[44, 22, 14],
[45, 23, 9],
[46, 24, 9],
[47, 25, 9],
[48, 25, 9],
[49, 26, 8],
[50, 27, 8],
[51, 28, 8],
[52, 28, 8],
[53, 29, 12],
[54, 29, 12],
[55, 30, 23],
[56, 31, 14],
[57, 32, 14],
[58, 33, 14],
[59, 34, 23],
[60, 35, 12],
[61, 36, 12],
[62, 37, 12],
[63, 38, 13],
[64, 39, 26],
[65, 40, 15],
[66, 41, 15],
[67, 41, 15],
[68, 41, 15],
[69, 42, 15],
[70, 42, 15],
[71, 43, 15],
[72, 44, 16],
[73, 45, 23],
[74, 46, 22],
[75, 47, 11],
[76, 48, 11],
[77, 49, 18],
[78, 50, 18],
[79, 51, 18],
[80, 52, 6],
[81, 53, 19],
[82, 53, 19],
[83, 54, 7],
[84, 55, 20],
[85, 56, 4],
[86, 57, 21],
[87, 58, 23],
[88, 59, 23],
[89, 59, 23],
[90, 60, 23],
[91, 61, 17],
[92, 62, 25],
[93, 63, 27],
[94, 64, 27],
[95, 65, 28],
[96, 66, 10],
[97, 67, 24],
[98, 68, 29],
[99, 69, 29],
[100, 70, 30]
]


trees_order_to_family = [
[1, 1, 2, 3, 4, 5, 6], 
[2, 7], 
[3, 8, 9], 
[4, 12, 56], 
[5, 13, 14, 15, 16, 17, 18, 19, 20], 
[6, 52], 
[7, 10, 11, 54], 
[8, 26, 27, 28], 
[9, 23, 24, 25], 
[10, 66], 
[11, 47, 48], 
[12, 29, 35, 36, 37], 
[13, 38], 
[14, 22, 31, 32, 33], 
[15, 40, 41, 42, 43], 
[16, 44], 
[17, 61], 
[18, 49, 50, 51], 
[19, 53], 
[20, 55], 
[21, 21, 57], 
[22, 46], 
[23, 30, 34, 45, 58, 59, 60], 
[24, 67], 
[25, 62], 
[26, 39], 
[27, 63, 64], 
[28, 65], 
[29, 68, 69], 
[30, 70]]


trees_family_to_species = [
[1, 1], 
[2, 2], 
[3, 3, 4, 5, 6], 
[4, 7, 8], 
[5, 9, 10, 11, 12], 
[6, 13], 
[7, 14], 
[8, 15], 
[9, 16], 
[10, 17, 18], 
[11, 19], 
[12, 20], 
[13, 21], 
[14, 22], 
[15, 23], 
[16, 24, 25, 26, 27, 28, 29, 30, 31], 
[17, 32, 33, 34, 35], 
[18, 36, 37], 
[19, 38, 39, 40], 
[20, 41, 42], 
[21, 43], 
[22, 44], 
[23, 45], 
[24, 46], 
[25, 47, 48], 
[26, 49], 
[27, 50], 
[28, 51, 52], 
[29, 53, 54], 
[30, 55], 
[31, 56], 
[32, 57], 
[33, 58], 
[34, 59], 
[35, 60], 
[36, 61], 
[37, 62], 
[38, 63], 
[39, 64], 
[40, 65], 
[41, 66, 67, 68], 
[42, 69, 70], 
[43, 71], 
[44, 72], 
[45, 73], 
[46, 74], 
[47, 75], 
[48, 76], 
[49, 77], 
[50, 78], 
[51, 79], 
[52, 80], 
[53, 81, 82], 
[54, 83], 
[55, 84], 
[56, 85], 
[57, 86], 
[58, 87], 
[59, 88, 89], 
[60, 90], 
[61, 91], 
[62, 92], 
[63, 93], 
[64, 94], 
[65, 95], 
[66, 96], 
[67, 97], 
[68, 98], 
[69, 99], 
[70, 100]
]


trees_order_to_species = [
[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
[2, 14], 
[3, 15, 16], 
[4, 20, 85], 
[5, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], 
[6, 80], 
[7, 17, 18, 19, 83], 
[8, 49, 50, 51, 52], 
[9, 45, 46, 47, 48], 
[10, 96], 
[11, 75, 76], 
[12, 53, 54, 60, 61, 62], 
[13, 63], 
[14, 44, 56, 57, 58], 
[15, 65, 66, 67, 68, 69, 70, 71], 
[16, 72], 
[17, 91], 
[18, 77, 78, 79], 
[19, 81, 82], 
[20, 84], 
[21, 43, 86], 
[22, 74], 
[23, 55, 59, 73, 87, 88, 89, 90], 
[24, 97], 
[25, 92], 
[26, 64], 
[27, 93, 94], 
[28, 95], 
[29, 98, 99], 
[30, 100]
]

def get_family(order):
    order_to_family = {order[0]: [num - 1 for num in order[1:]] for order in trees_order_to_family}
    return order_to_family.get(order, [])

def get_species(family):
    family_to_species = {family[0]: [num - 1 for num in family[1:]] for family in trees_family_to_species}
    return family_to_species.get(family, [])

def from_order_get_species(order):
    order_to_species = {order[0]: [num - 1 for num in order[1:]] for order in trees_order_to_species}
    return order_to_species.get(order, [])

def get_classnames(filename):
    name_list = []

    with open(filename, "r") as file:
        for line in file:
            name = line.strip()  # 去除行尾的换行符和空格
            name_list.append(name)
    return name_list

def get_order_family_target(targets):


    order_target_list = []
    family_target_list = []


    for i in range(targets.size(0)):

        order_target_list.append(trees[targets[i]][2]-1)
        family_target_list.append(trees[targets[i]][1]-1)



    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())   
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())   

    return order_target_list, family_target_list
