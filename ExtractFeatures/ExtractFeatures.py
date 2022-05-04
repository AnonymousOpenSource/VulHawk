# -*-coding:utf-8-*-

import idaapi
import idc
import pickle
import os
from idaapi import get_func
import ida_kernwin as kw
import ida_hexrays as hr
import ida_funcs
import ida_range
import re
from micro_lib import maturity_levels, traverse_minsn, LEVEL


idaapi.auto_wait()

file_name = os.path.splitext(idc.get_idb_path())[0]
functions = []
binary_name = idaapi.get_input_file_path()


def get_MircoGraph_func(ea, level, comment=False):
    pfn = ida_funcs.get_func(ea)
    sea = pfn.start_ea
    eea = pfn.end_ea

    fn_name = ida_funcs.get_func_name(pfn.start_ea)
    text, mmat = maturity_levels[level]
    mba_flags = 0x400000  # No comments
    if comment:
        mba_flags = 0

    mbr = hr.mba_ranges_t()
    hf = hr.hexrays_failure_t()
    ml = hr.mlist_t()
    mbr.ranges.push_back(ida_range.range_t(sea, eea))
    mba = hr.gen_microcode(mbr, hf, ml, hr.DECOMP_WARNINGS, mmat)
    if not mba:
        return None
    mba.set_mba_flags(mba_flags)

    function_dict = {}

    function_dict['name'] = fn_name
    function_dict['blocks'] = list()
    function_dict['filename'] = binary_name
    s = get_bytes(sea, eea - sea)
    function_dict['bytes'] = "".join("{:02x}".format(c) for c in s)

    cur = 1

    while True:
        block_dict = {}
        cur_block = mba.get_mblock(cur)
        if cur_block.empty():
            break
        block_dict['id'] = cur - 1
        block_dict['succs'] = [_ for _ in cur_block.succset]
        tm = traverse_minsn(cur_block)
        cur += 1
        block_dict["minsn"] = tm.minsn
        block_dict["mtoken"] = tm.mtoken
        function_dict['blocks'].append(block_dict)
    return function_dict


for seg_ea in Segments():
    if idc.get_segm_name(seg_ea) != ".text" and idc.get_segm_name(seg_ea) != "LOAD":
        continue
    for function_ea in Functions(get_segm_start(seg_ea), get_segm_end(seg_ea)):
        function = get_MircoGraph_func(function_ea, LEVEL)

        if function:
            functions.append(function)

pickle.dump(functions, open("{}.pkl".format(file_name), "wb"))
idc.qexit(0)
