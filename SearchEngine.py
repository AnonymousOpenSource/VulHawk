# -*-coding:utf-8-*-
import time
from pathlib import Path
import pickle
import torch
from tqdm import tqdm
import numpy as np
import os
import heapq
from pathlib import Path
import os
from utils.libs import S2VGraph, read_pickle, BlockEmbedding, ControlGraphEmbedding
from utils.FunctionNet import Net, MyData, AdapterHead
from Tokenizer.InstructionTokenizer import InstructionTokenizer
from torch_geometric.data import DataLoader
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def build_for_one_list(inputs=[], input_type="file", min_blk=3, ckpt=None, FEconfig=None):
    FunctionName = []
    VectorTable = []
    DetailData = {}
    FunctionMap = {}
    global CfgEmbedding
    id = 0

    input_loader = DataLoader([MyData(_.node_features, _.edge_mat, torch.tensor(_.entropy)) for _ in inputs], batch_size=512)
    pbar = tqdm(input_loader)
    representations = []
    for batch_func in pbar:
        batch_output = CfgEmbedding.generate(batch_func)
        representations.extend(batch_output)

    for i, func in enumerate(inputs):
        funcName = func.label
        funcName = func.binary_name + os.path.sep + funcName
        funcName = funcName.replace('F:\\Code\\SoCan\\experiments\\', '')
        if len(func) < min_blk:
            continue

        func_features = representations[i].cpu()
        FunctionName.append((id, funcName))
        DetailData[id] = {"binary_name": func.binary_name,
                          "funcname": funcName
                          }
        FunctionMap[funcName] = id
        id += 1
        VectorTable.append(func_features)



    return FunctionName, VectorTable, FunctionMap



def extract_basic_block_from_binary(input_files=[], min_blk=3):
    global BlkEmbedding
    data = []
    pbar = tqdm(input_files)
    filter_suffix = (".i64", "idb", ".pkl", ".dat", ".id0", "id1", "id2", ".nam", ".til", ".a")
    for fname in pbar:
        if os.path.isdir(fname):
            continue
        if fname.endswith(filter_suffix):
            continue
        get_idb_file(fname, mode=64)
        fname = fname + ".pkl"
        func_dict = read_pickle(fname, BlkEmbedding=BlkEmbedding, dim=64, min_blk=min_blk)
        data.extend(list(func_dict.values()))
    return data


def get_idb_file(binary_name, mode=64):
    print("[+] Extracting Binary Features")
    if mode == 64:
        os.system('start {} -A -S"{}" "{}"'.format(cmd64, script, binary_name))
    elif mode == 32:
        os.system('start {} -A -S"{}" "{}"'.format(cmd32, script, binary_name))
    while True:
        if os.path.exists(binary_name+".pkl"):
            time.sleep(0.5)
            break

    print("[+] Extraction is complete!")



def build_repository(min_blk=3):
    filelist = []
    filelist += [str(x) for x in Path("demo/").glob("**/*")]
    file_dict = extract_basic_block_from_binary(input_files=filelist, min_blk=min_blk)
    FunctionName, VectorTable, FunctionMap = build_for_one_list(file_dict, min_blk=min_blk)
    pickle.dump((FunctionName, VectorTable, FunctionMap), open("repository.dat.pkl", "wb"))

if __name__ == '__main__':
    cmd64 = "D:\\Programs\\IDA7.5SP3\\ida64.exe"
    cmd32 = "D:\\Programs\\IDA7.5SP3\\ida.exe"
    script = "F:\\Code\\VulHawk\\ExtractFeatures\\ExtractFeatures.py"

    model_path = "GraphEmbedding/checkpoint"
    tokenizer_path = "Tokenizer/model_save/tokenizer.model"
    blkEmbedding = "./BlockEmbedding/checkpoint"
    device = torch.device('cuda:0')
    dimension = 64
    BlkEmbedding = BlockEmbedding(model_directory=blkEmbedding, tokenizer=tokenizer_path, batch_size=60, device=device)
    CfgEmbedding = ControlGraphEmbedding(pretrained_model=model_path)
    build_repository()

