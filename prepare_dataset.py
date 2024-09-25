import distorm3
import sys
import random
import numpy as np
import networkx as nx
import pandas as pd
import json
import re
import glob
import os
from stellargraph.data.explorer import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec

re_regs = re.compile(
    "AL|AH|AX|EAH|EAX|RAH|RAX|BL|BH|BX|EBX|RBH|RBX|CL|CH|CX|ECX|RCX|DL|DH"
    "|DX|EDL|EDH|EDX|RDX|SI|ESI|RSI|DI|EDI|RDI|SP|ESP|RSP|BP|EBP|RBP|IP|EIP"
    "|RIP|R8|R9|R10|R11|R12|R13|R14|R15|R8D|R9D|R10D|R11D|R12D|R13D|R14D|"
    "R15D|R8W|R9W|R10W|R11W|R12W|R13W|R14W|R15W|XMM0|XMM1|XMM2|XMM3|XMM4|XMM5|XMM6|XMM7")

re_seg = re.compile("CS|DS|ES|FS|GS|SS")
pseudo_inst = re.compile("DB|DW|DD|DQ|DT|DDQ|DO|RESB|RESW|RESD|RESQ|REST|RESDDQ|RESO|EQU|INCBIN|TIMES")
re_const = re.compile("0x[0-9a-fA-F]+|\d+")
addr = re.compile("\[[^\]]*\]")
b_addr  = re.compile("BYTE\s\[[^\]]*\]")
w_addr  = re.compile("WORD\s\[[^\]]*\]")
d_addr  = re.compile("DWORD\s\[[^\]]*\]")
q_addr  = re.compile("QWORD\s\[[^\]]*\]")

def diassembler(filename, bits='32bit'):
    mode = distorm3.Decode32Bits
    offset = 0
    with open(filename, 'rb') as f:
        data = f.read()

    i = distorm3.DecodeGenerator(offset, data, mode)
    return i

def regex_check(data):
    if(len(b_addr.findall(data))):
        return 'BYTE_MEM'
    elif(len(d_addr.findall(data))):
        return 'DWORD_MEM'
    elif(len(q_addr.findall(data))):
        return 'QWORD_MEM'
    elif(len(w_addr.findall(data))):
        return 'WORD_MEM'
    elif(len(addr.findall(data))):
        return 'MEM'
    elif(len(re_const.findall(data))):
        return 'CONST'
    elif(len(re_regs.findall(data))):
        return 'REG'
    elif(len(re_seg.findall(data))):
        return re_seg.findall(data)[0]

    else:
        return data


def func(inst, opcode):
    tmp = inst.split(' ', 1)
    while '' in tmp:
        tmp.remove('')
    opcode = tmp[0].strip()
    pseudo_ins = pseudo_inst.match(opcode)
    if(pseudo_ins):
        return pseudo_ins.group()
    src = ''
    dest = ''
    if(len(tmp)>1):
        d = tmp[1].split(',')
        if(len(d)>1):
            src = regex_check(d[1].strip())
        dest = regex_check(d[0].strip())
    if(src == None):
        return opcode.strip()+' '+dest.strip()
    elif(dest == None):
        return opcode.strip()
    else:
        return opcode.strip()+' '+dest.strip()+' '+src.strip()

def getOpcode(filename, delimeter='\n', bits='32bit'):

    iterable = diassembler(filename,bits)

    opcode_code = ''
    samp_code = ''
    for (offset, size, instruction, hexdump) in iterable:

        # To avoid TypeError: a bytes-like object is required, not 'str'
        instruction = instruction
        o_data = func(instruction, hexdump)
        samp_code += o_data+'\n'
        opcode = instruction.split(" ")[0]  # get opcode
        opcode_code += opcode+delimeter

    return opcode_code, samp_code


def unique(l, operands):
    unique_features=operands
    try:
        for i in range(1, len(l)):
            if(l[i] not in unique_features):
                unique_features[l[i]] = len(unique_features)
        return unique_features
    except:
        pass

def inst_to_num(data):
    oset = set()
    content = data.split('\n')
    oset.update(set(content))
    if('' in oset):
        oset.remove('')
    c = 0
    dic = {}
    for i in oset:
        dic[i.strip()] = c
        c = c+1

    return dic

# Reads the file 

def readfile(filename):
    delimiter = ','
    content, samp_code = getOpcode(filename, delimiter)
    opcode_file = filename.split('.')[0]+'.opcode_seq'
    with open(opcode_file, 'w') as f:
        f.write(samp_code)


    temp = filename.split('.', 1)[0]+'_node_mapping.json'
    with open(opcode_file, 'r') as f:
        data = f.read()
    dict_op = inst_to_num(data)
    node_feature = {}
    operands = {'Null':0}
    for i in dict_op:
        empty= ['Null']*3
        t = i.strip().split()
        operands.update(unique(t, operands))
        if(len(t)<=3):
            for x in range(len(t)):
                empty[x] = t[x]
            node_feature[str(dict_op[i])] = {'dest':empty[1], 'src':empty[2]}
        else:
            print(t)

    #print(operands)
    with open(temp, 'w') as f:
        json.dump(node_feature, f)
    try:
        with open('node_feature_mapping.json', 'r') as infile:
            feature_map = json.load(infile)
    except:
        feature_map = {}
        print("Node Feature Mapping File is not created")
    with open('node_feature_mapping.json', 'w') as outfile:
        for i in operands:
            if(i not in feature_map):
                feature_map[i] = len(feature_map)

        json.dump(feature_map, outfile)



    try:
        with open('all_inst_mapping.json', 'r') as infile:
            inst_map = json.load(infile)
    except:
        inst_map = {}
        print("Intruction mapping is not created")
    with open('all_inst_mapping.json', 'w') as outfile:
        for i in dict_op:
            if(i not in inst_map):
                inst_map[i] = len(inst_map)+1

        json.dump(inst_map, outfile)



    content = data.split('\n')
    if('' in content):
        content.remove('')
    curr_op = content[0].strip()
    edges = {}
    for i in range(1,len(content)):
        nxt_op = content[i].strip()
        cell = str(dict_op[curr_op])+ '->' +str(dict_op[nxt_op])
        if(cell in edges):
            edges[cell] += 1
        else:
            edges[cell] = 1
        curr_op = nxt_op
    edge_file = opcode_file.replace('.opcode_seq','.edge')
    with open(edge_file, 'w') as f:
        for i in edges:
            f.write(i+' '+str(edges[i])+'\n')

    tmp = {}
    for i in edges:
        t = i.split('->')
        if(t[0] not in tmp):
            tmp[t[0]]= []
        tmp[t[0]].append(tuple((t[1], edges[i])))


    #Graph Creation
    edges_from = []
    edges_to = []
    weights = []
    #Edges
    for i in edges:
        t = i.split('->')
        edges_from.append(t[0])
        edges_to.append(t[1])
        weights.append(edges[i])

    g_edges = pd.DataFrame({"source":edges_from, "target":edges_to, "weights":weights})
    #Nodes
    index = []
    dest = []
    src = []
    for i in node_feature:
        index.append(i)
        dest.append(operands[node_feature[i]['dest']])
        src.append(operands[node_feature[i]['src']])

    g_nodes = pd.DataFrame({"dest":dest, "src":src}, index=index)
    G = StellarGraph(g_nodes, g_edges)
    return G

def main():
    graph = readfile(sys.argv[1])


if __name__ == "__main__":
    main()

