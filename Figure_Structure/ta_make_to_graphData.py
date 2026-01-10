import os
import pickle
from transformers import RobertaTokenizer
import torch
from torch_geometric.data import Data
import xml.etree.ElementTree as ET
import numpy as np
import random


def traverse_graph_direct(graph):
    print("Vertices:")
    for vertex in graph.vs:
        print("Node ID: %s" % vertex['name'])
        print("Node code: %s" % vertex['code'])
        print("Node filepath: %s" % vertex['filepath'])
        print("Node functionId: %s" % vertex['functionId'])
        print("Node location: %s" % vertex['location'])
        print("Node type: %s" % vertex['type'])
        print('\n')
    print(len(graph.vs))

    print("\nEdges:")

    for edge in graph.es:
        source = edge.source
        target = edge.target
        print("Edge from %s to %s, Type: %s" % (graph.vs[source]['name'], graph.vs[target]['name'], edge['type']))
    print(len(graph.es))


def prepare_graph_data(cpg, tokenizer, graph_label=None):
    node_features = []
    node_types = []

    all_node_types = list(set(v['type'] for v in cpg.vs))
    type_to_id = {t: i for i, t in enumerate(all_node_types)}
    max_length = 128  

    for node in cpg.vs:
        code = node['code'] or ""
        node_type = node['type'] or "UNKNOWN"

        tok = tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tok['input_ids'].squeeze(0) 

        type_id = type_to_id.get(node_type, 0)
        node_types.append(type_id)

        node_features.append(input_ids)

    x = torch.stack(node_features, dim=0)  
    node_type_tensor = torch.tensor(node_types, dtype=torch.long) 

    edge_index = []
    edge_attr = []

    edge_type_vocab = {
        "AST": 0, "CONTROL": 1, "DATA": 2,
        "CALL": 3, "FLOW": 4, "ENTRY": 5
    }

    default_edge_type = 0

    for e in cpg.es:
        src, tgt = e.source, e.target
        edge_type = e['type'] if 'type' in e.attributes() else "UNKNOWN"

        etype = edge_type_vocab.get(edge_type, default_edge_type)

        edge_index.append([src, tgt])
        edge_attr.append([etype]) 

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[default_edge_type]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    y = torch.tensor([graph_label], dtype=torch.long) if graph_label is not None else None

    return Data(
        x=x,
        node_type=node_type_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(cpg.vs)
    )


def find_all_graph_label():
    vul_line = '../datasets/target_vul.xml'
    path_data = '/home/zzb/next_model/datasets/'
    tree = ET.parse(vul_line)
    root = tree.getroot()

    vul_info = {}
    for testcase in root.findall('testcase'):
        CWE_ID = testcase.attrib['CWE_ID']
        target = testcase.attrib['target']
        target = int(target)

        file_element = testcase.find('file')
        file_path = file_element.attrib.get('path')
        merge_file_path = os.path.join(path_data, file_path)
        m_l = merge_file_path.split('\\')
        merge_file_path = "/".join(m_l)

        flaw_element = file_element.find('flaw')
        flaw_line = flaw_element.attrib['line']
        flaw_line = flaw_line.strip()[1:-1]
        flaw_line = flaw_line.split(', ')
        lines = [int(line) for line in flaw_line]

        if target == 1:
            if len(lines) == 1:
                if lines[0] == -1: i in range(1, 3000)]

        if merge_file_path not in vul_info.keys():
            vul_info[merge_file_path] = lines
        else:
            old_lines = vul_info[merge_file_path]
            for i in lines:
                old_lines.append(i)
            vul_info[merge_file_path] = old_lines

    return vul_info


def find_graph_label(cpg, vul_info):
    label = 0
    file_path = None

    for v in cpg.vs:
        if v['filepath']:
            file_path = v['filepath']
            break

    if not file_path or file_path not in vul_info:
        return 0

    vul_lines = vul_info[file_path]
    for v in cpg.vs:
        location = v['location']
        if location is None:
            continue

        if isinstance(location, str):
            if ':' in location:
                row = location.split(':')[0]
                if row.isdigit():
                    row = int(row)
                    if row in vul_lines:
                        label = 1
                        break
            else:
                if location.isdigit():
                    row = int(location)
                    if row in vul_lines:
                        label = 1
                        break
        elif isinstance(location, int):
            if location in vul_lines:
                label = 1
                break

    return label, file_path


def save_graph_datas(cpg_path, tokenizer_path):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    vul_info = find_all_graph_label()

    graph_datas = []
    labels = []
    locations = []
    for root, dirs, files in os.walk(cpg_path):
        dirs.sort()  
        files.sort() 

        for file in files:
            cpg_file_path = os.path.join(root, file)
            with open(cpg_file_path, 'rb') as fin:
                cpg = pickle.load(fin)

            graph_label, location = find_graph_label(cpg, vul_info)
            graph_data = prepare_graph_data(cpg, tokenizer, graph_label)
            graph_datas.append(graph_data)

            labels.append(graph_label)
            locations.append(location)

    with open("java_cpg_graph/graph_datas.pkl", "wb") as f_out:
        pickle.dump(graph_datas, f_out)
    with open('java_cpg_graph/labels.pkl', "wb") as f_out:
        pickle.dump(labels, f_out)
    with open('java_cpg_graph/locations.pkl', "wb") as f_out:
        pickle.dump(locations, f_out)

    for lo in labels:
        with open('java_cpg_graph/labels.txt', 'a') as txt_file:
            txt_file.write(str(lo)+'\n')
            
    for lo in locations:
        with open('java_cpg_graph/locations.txt', 'a') as txt_file:
            txt_file.write(str(lo)+'\n')


def main():
    cpg_path = "java_cpg"
    tokenizer_path = '../Sequence_AST/unixcoder-base'

    save_graph_datas(cpg_path, tokenizer_path)


if __name__ == "__main__":
    main()
