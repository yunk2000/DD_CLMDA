import pickle

import javalang
import json
import os
from igraph import *
node_id_counter = 0

def traverse_graph_direct(graph):
    print("Vertices:")
    for vertex in graph.vs:
        print("Node ID: %s" % vertex['name'])
        print("Node type: %s" % vertex['type'])
        print("Node code: %s" % vertex['code'])
        print("Node filepath: %s" % vertex['filepath'])
        print("Node location: %s" % vertex['location'])
    print(len(graph.vs))

    print("\nEdges:")

    for edge in graph.es:
        source = edge.source
        target = edge.target
        print("Edge from %s to %s, Type: %s" % (graph.vs[source]['name'], graph.vs[target]['name'], edge['type']))
    print(len(graph.es))

def handle_value(value):
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def generate_node_id():
    global node_id_counter
    node_id_counter += 1
    return node_id_counter

def traverse_ast(node, g, file_path, parent_Node=None):
    node_id = generate_node_id()
    node_id = str(node_id)

    node_info = {
        'type': type(node).__name__,
        'location': node.position.line if hasattr(node, 'position') and node.position else None,
        'column': node.position.column if hasattr(node, 'position') and node.position else None,
        'code': handle_value(getattr(node, 'value', str(node))),
        'filepath': os.path.abspath(file_path),
    }
    g.add_vertex(node_id, **node_info)

    if parent_Node is not None:
        edge_prop = {'type': 'IS_AST_PARENT', 'var': None, 'curved': False}
        g.add_edge(parent_Node, node_id, **edge_prop)

    if isinstance(node, javalang.ast.Node):
        for field in node.attrs:
            value = getattr(node, field)
            if isinstance(value, (list, tuple)):
                for item in value:
                    g = traverse_ast(item, g, file_path, parent_Node=node_id)
            elif isinstance(value, javalang.ast.Node):
                g = traverse_ast(value, g, file_path, parent_Node=node_id)

    elif isinstance(node, list):
        for item in node:
            g = traverse_ast(item, g, file_path, parent_Node=node_id)

    return g

def isEdgeExists(ast, startnode, endnode):
    for edge in ast.es:
        if ast.vs[edge.tuple]['name'][0] == startnode and ast.vs[edge.tuple]['name'][1] == endnode:
            return True
        else:
            continue
    return False

def isDelASTNode(v):
    if v['code'] == 'EXIT':
        return False
    elif v['location'] == None and v['type'] != 'FunctionDef':
        return True
    else:
        return False

def addASTEdge(ast, startnode, endnode):
    if isEdgeExists(ast, startnode, endnode):
        return ast

    edge_prop = {'type': 'IS_AST_PARENT', 'var': None, 'curved': False}
    ast.add_edge(startnode, endnode, **edge_prop)
    return ast

def deleteEdgeOfAST(ast):
    vertices_to_delete = []
    for vertex in ast.vs:
        if isDelASTNode(vertex):
            list_pre = vertex.predecessors()
            list_su = vertex.successors()

            if len(list_su) != 0:
                for i in range(len(list_pre)):
                    for j in range(len(list_su)):
                        addASTEdge(ast, list_pre[i], list_su[j])

            vertices_to_delete.append(vertex.index)

    ast.delete_vertices(vertices_to_delete)
    return ast

def draw_graph(graph, output_file="ast_graph.png"):
    
    layout = graph.layout("fr")
    visual_style = {
        "vertex_size": 20,
        "vertex_label": graph.vs["name"],
        "vertex_label_size": 10,
        "bbox": (1500, 1500),
        "margin": 100,
    }
    plot(graph, output_file, layout=layout, **visual_style)


def main():
    java_path = '/home/zzb/next_model/datasets/targetVul_java/'
    javaAst_output_path = './java_ast'
    java_ast_png_path = 'java_ast_png'

    for root, dirs, files in os.walk(java_path):
        dirs.sort()
        files.sort()

        for file in files:
            java_ast_path = os.path.join(root, file)
            cwe_id = java_ast_path.split('/')[-2]

            javaAst_cwe_path = os.path.join(javaAst_output_path, cwe_id)
            if not os.path.exists(javaAst_cwe_path):
                os.mkdir(javaAst_cwe_path)
            java_png_cwe_path = os.path.join(java_ast_png_path, cwe_id)
            if not os.path.exists(java_png_cwe_path):
                os.mkdir(java_png_cwe_path)

            with open(java_ast_path, 'r', encoding='utf-8') as file:
                java_code = file.read()

            java_file = java_ast_path.split('/')[-1]
            java_ast_file = java_file.split('.')[:-1]
            java_ast_file_1 = '.'.join(java_ast_file)
            java_ast_png_path_file = os.path.join(java_png_cwe_path, f"{java_ast_file_1}_ast.png")
            path = os.path.join(javaAst_cwe_path, java_ast_file_1)
            if os.path.exists(path):
                continue
            
            tree = javalang.parse.parse(java_code)
            g = Graph(directed=True)

            java_ast = traverse_ast(tree, g, java_ast_path)

            ast = deleteEdgeOfAST(java_ast)
            draw_graph(ast, output_file=java_ast_png_path_file)

            with open(path, 'wb') as f:
                pickle.dump(ast, f)

if __name__ == '__main__':
    main()
