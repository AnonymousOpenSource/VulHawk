# one-to-one comparison

import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

archs = ["x86"]
compilers = ["clang-11", "gcc-10"]
optimization = ["O0", "O1", "O2", "O3", "Os", "Ofast"]

FunctionName, VectorTable, FunctionMap = pickle.load(open("repository.dat.pkl", "rb"))


def main():
    keys = np.random.randint(0, len(FunctionMap), 8000000)
    type_dict = {"clang-11": {
        "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "arm-32-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "arm-64-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "mips-32-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        }
    }

    for i in FunctionName:
        namesplit = i[1].split(os.path.sep)
        compiler = namesplit[0]
        opt = namesplit[2]
        if compiler not in type_dict:
            continue
        type_dict[compiler][opt].append(i[0])
    random_type_dict = {"clang-11": {
        "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "arm-32-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "arm-64-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        },
        "mips-32-gcc-10": {
            "O0": [], "O1": [], "O2": [], "O3": [], "Os": [], "Ofast": []
        }
    }

    for compiler in random_type_dict:
        for opt in random_type_dict[compiler]:
            if len(type_dict[compiler][opt]) > 0:
                random_type_dict[compiler][opt] = list(np.random.randint(0, len(type_dict[compiler][opt]), 200000))



    # XC
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0
    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]
        if "O0" not in namesplit[2] and "O3" not in namesplit[2]:
            continue
        if "gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "clang-11")
            random_compiler = "clang-11"
        elif "clang-11" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "gcc-10")
            random_compiler = "gcc-10"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)
        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][namesplit[2]].pop()
            random_key = type_dict[random_compiler][namesplit[2]][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)

        N += 1
        if N >= N_Max:
            break
    print(N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))


    # XO
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0

    opt_random_keys = list(np.random.randint(0, 5, 200000))

    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        if namesplit[0] != "clang-11" and namesplit[0] != "gcc-10":
            continue
        anchor_vector = VectorTable[key]
        cur_opt = ["O0", "O1", "O2", "O3", "Os", "Ofast"]
        cur_opt.remove(namesplit[2])
        opt_random_key = opt_random_keys.pop()
        random_opt = cur_opt[opt_random_key]
        positive_sample_name = FunctionName[key][1].replace(namesplit[2], random_opt)
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)
        for n in range(n_negtive):
            random_key_index = random_type_dict[namesplit[0]][random_opt].pop()  # fixed compiler
            random_key = type_dict[namesplit[0]][random_opt][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)
        N += 1
        if N >= N_Max:
            break
    print(N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))




    # XA
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0
    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]

        if "gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "arm-64-gcc-10")
            random_compiler = "arm-64-gcc-10"
        elif "arm-64-gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "gcc-10")
            random_compiler = "gcc-10"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)
        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][namesplit[2]].pop()  # fixed opt
            random_key = type_dict[random_compiler][namesplit[2]][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)
        N += 1
        if N >= N_Max:
            break
    print(N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))

    # XC+XO
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0
    opt_random_keys = list(np.random.randint(0, 5, 200000))
    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]
        cur_opt = ["O0", "O1", "O2", "O3", "Os", "Ofast"]
        try:
            cur_opt.remove(namesplit[2])
        except:
            print(namesplit)
            exit()
        opt_random_key = opt_random_keys.pop()
        random_opt = cur_opt[opt_random_key]
        if "gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "clang-11").replace(namesplit[2],
                                                                                                  random_opt)
            random_compiler = "clang-11"
        elif "clang-11" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "gcc-10").replace(namesplit[2],
                                                                                                random_opt)
            random_compiler = "gcc-10"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)

        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][random_opt].pop()
            random_key = type_dict[random_compiler][random_opt][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)
        N += 1
        if N >= N_Max:
            break
    print("XC+XO", N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))


    # XO+XA
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0
    opt_random_keys = list(np.random.randint(0, 3, 500000))
    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]
        cur_opt = ["O0", "O1", "O2", "O3"]
        if namesplit[2] not in cur_opt:
            continue
        cur_opt.remove(namesplit[2])
        opt_random_key = opt_random_keys.pop()
        random_opt = cur_opt[opt_random_key]
        if "gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "arm-64-gcc-10").replace(namesplit[2],
                                                                                                  random_opt)
            random_compiler = "arm-64-gcc-10"
        elif "arm-64-gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "gcc-10").replace(namesplit[2],
                                                                                                random_opt)
            random_compiler = "gcc-10"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)

        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][random_opt].pop()
            random_key = type_dict[random_compiler][random_opt][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)
        N += 1
        if N >= N_Max:
            break
    print("XO+XA", N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))


    # XA+XC
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0

    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]

        if "clang-11" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "arm-64-gcc-10")
            random_compiler = "arm-64-gcc-10"
        elif "arm-64-gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "clang-11")
            random_compiler = "clang-11"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)
        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][namesplit[2]].pop()
            random_key = type_dict[random_compiler][namesplit[2]][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)

        N += 1
        if N >= N_Max:
            break
    print("XA+XC", N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))


    # XM
    Y = []
    Y_pred = []
    n_negtive = 1
    N_Max = 50000
    N = 0
    opt_random_keys = list(np.random.randint(0, 3, 500000))
    for key in keys:
        namesplit = FunctionName[key][1].split(os.path.sep)
        anchor_vector = VectorTable[key]
        cur_opt = ["O0", "O1", "O2", "O3"]
        if namesplit[2] not in cur_opt:
            continue
        cur_opt.remove(namesplit[2])
        opt_random_key = opt_random_keys.pop()
        random_opt = cur_opt[opt_random_key]
        if "clang-11" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "arm-64-gcc-10").replace(namesplit[2], random_opt)
            random_compiler = "arm-64-gcc-10"
        elif "arm-64-gcc-10" == namesplit[0]:
            positive_sample_name = FunctionName[key][1].replace(namesplit[0], "clang-11").replace(namesplit[2], random_opt)
            random_compiler = "clang-11"
        else:
            continue
        if positive_sample_name not in FunctionMap:
            continue
        positive_sample_vector = VectorTable[FunctionMap[positive_sample_name]]
        Y_pred.append(anchor_vector.dist(positive_sample_vector).numpy())
        Y.append(1)
        for n in range(n_negtive):
            random_key_index = random_type_dict[random_compiler][random_opt].pop()
            random_key = type_dict[random_compiler][random_opt][random_key_index]

            neg_sample_vector = VectorTable[random_key]
            neg_sample_name = FunctionName[random_key][1]
            Y_pred.append(anchor_vector.dist(neg_sample_vector).numpy())
            Y.append(0)
        N += 1
        if N >= N_Max:
            break
    print("XM", N)
    Y_similarity = 1 / (1 + np.array(Y_pred))
    print(roc_auc_score(Y, Y_similarity))

np.random.seed(966)
main()
