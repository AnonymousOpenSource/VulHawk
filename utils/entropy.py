import os
import math
import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm
def eta(data, unit='natural'):
    base = {
        'shannon': 2.,
        'natural': math.exp(1),
        'hartley': 10.
    }
    if len(data) <= 1:
        return 0
    counts = Counter()
    for d in data:
        counts[d] += 1
    ent = 0
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])
    return ent

def raw_byte_eta(raw_file):
    f = open(raw_file, "rb")
    raw_byte = f.read()
    f.close()
    return eta([i for i in raw_byte])

def func_raw_byte_eta(input_pkl):

    func_list = input_pkl
    all_bytes = ""
    for func in func_list:
        funcName = func['name']
        all_bytes += func['bytes']
    return eta(["".join(_) for _ in zip(all_bytes[::2], all_bytes[1::2])])

def func_opcode_eta(input_pkl):
    func_list = input_pkl
    all_opcode = []
    for func in func_list:
        funcName = func['name']
        func_opcode = []
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            if "minsn" not in blk:
                continue
            func_opcode += [_[0] for _ in blk["minsn"] if _]
        all_opcode += func_opcode
    return eta(all_opcode)

def func_opcode_oprand_raw_eta(input_file):
    pkl_name = input_file + ".pkl"
    func_list = pickle.load(open(pkl_name, "rb"))
    all_opcode = []
    for func in func_list:
        funcName = func['name']
        func_opcode = []
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            if "minsn" not in blk:
                continue
            func_opcode += ["".join(_) for _ in blk["minsn"] if _]
        all_opcode += func_opcode
    return eta(all_opcode)

def func_opcode_oprand_eta(input_pkl):
    func_list = input_pkl
    all_opcode = []
    for func in func_list:
        funcName = func['name']
        func_opcode = []
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            if "minsn" not in blk:
                continue
            func_opcode += [_[0] + "".join([__ for __ in _[1:] if len(__)<20]) for _ in blk["minsn"] if _]
        all_opcode += func_opcode
    return eta(all_opcode)

def func_opcode_oprand_token_eta(input_pkl):
    func_list = input_pkl
    all_opcode = []
    for func in func_list:
        funcName = func['name']
        func_opcode = []
        for cur, blk in zip(range(len(func['blocks'])), func['blocks']):
            if "mtoken" not in blk:
                continue
            func_opcode += [_[0] + "".join([__ for __ in _[1:] if len(__) < 20]) for _ in blk["mtoken"] if _]
        all_opcode += func_opcode

    return eta(all_opcode)

def run(file_list):
    database = {}
    pbar = tqdm(file_list)
    for f in pbar:

        if os.path.isdir(f):
            continue
        filename = f.replace(".pkl", "")
        database[f.replace(".pkl", "")] = (raw_byte_eta(filename), func_raw_byte_eta(filename), func_opcode_eta(filename), func_opcode_oprand_eta(filename), func_opcode_oprand_raw_eta(filename))
    pickle.dump(database, open("entropy.pkl","wb"))

def traverse_file_eta(file_list, function_ptr):
    all_eta = []
    for f in file_list:
        if "." in f.split("\\")[-1]:
            continue
        if os.path.isdir(f):
            continue
        all_eta.append(function_ptr(f))
    print(sum(all_eta)/len(all_eta))



