import shutil
from joern.all import JoernSteps
from igraph import *
import os
import pickle

def traverse_graph_direct(graph):
    print("Vertices:")
    for vertex in graph.vs:
        print "Node ID: %s" % vertex['name']
        print "Node code: %s" % vertex['code']
        print "Node filepath: %s" % vertex['filepath']
        print "Node functionId: %s" % vertex['functionId']
        print "Node location: %s" % vertex['location']
        print "Node type: %s" % vertex['type']
        print '\n'
    print len(graph.vs)

    print "\nEdges:"

    for edge in graph.es:
        source = edge.source
        target = edge.target
        print "Edge from %s to %s, Type: %s" % (graph.vs[source]['name'], graph.vs[target]['name'], edge['var'])
    print len(graph.es)

def traverse_to_edge(results):
    for edge in results:
        start_node = edge.start_node
        end_node = edge.end_node
        print "Start Node Attributes:", start_node._id
        print "Start Node Attributes:", start_node.properties

        print "End Node Attributes:", end_node._id
        print "End Node Attributes:", end_node.properties

        print "Edge Attributes:", edge.type
        print "Edge Attributes:", edge
        print "-" * 50

def getALLFuncNode(db):
    query_str = "queryNodeIndex('type:Function')"
    results = db.runGremlinQuery(query_str)
    return results

def getFuncFile(db, func_id):
    query_str = "g.v(%d).in('IS_FILE_OF').filepath" % func_id
    ret = db.runGremlinQuery(query_str)
    return ret[0]

def isNodeExist(g, nodeName):
    if not g.vs:
        return False
    else:
        return nodeName in g.vs['name']

def getDirectChildren(db, func_id):
    query_str =  % (func_id)
    edges = db.runGremlinQuery(query_str)

    return edges

def sortedNodesByLoc(list_node):
    _list = []
    for node in list_node:
        if node['location'] == None:
            row = 'inf'
            col = 'inf'
        else:
            row, col = [int(node['location'].split(':')[0]), int(node['location'].split(':')[1])]
        _list.append((row, col, node))

    _list.sort(key=lambda x: (x[0], x[1]))
    list_ordered_nodes = [_tuple[2] for _tuple in _list]

    return list_ordered_nodes

def isDelASTNode(v):
    if v['code'] == 'EXIT':
        return False
    elif v['location'] == None and v['type'] != 'FunctionDef':
        return True
    elif v['code'] == '':
        return True
    else:
        return False

def isEdgeExists(ast, startnode, endnode):
    for edge in ast.es:
        if ast.vs[edge.tuple]['name'][0] == startnode and ast.vs[edge.tuple]['name'][1] == endnode:
            return True
        else:
            continue
    return False

def addASTEdge(ast, startnode, endnode):
    if isEdgeExists(ast, startnode, endnode):
        return ast

    edge_prop = {'var': 'IS_AST_PARENT'}
    ast.add_edge(startnode, endnode, **edge_prop)
    return ast

def drawGraph(db, edges, func_entry_node):
    g = Graph(directed=True)
    func_id = func_entry_node._id
    filepath = getFuncFile(db, func_id)

    for edge in edges:
        if edge.start_node.properties['code'] == 'ENTRY':
            startNode = str(edge.start_node.properties['functionId'])
        else:
            startNode = str(edge.start_node._id)

        if edge.start_node.properties['code'] == 'ERROR':
            continue

        if not isNodeExist(g, startNode):
            if edge.start_node.properties['code'] == 'ENTRY':
                node_prop = {
                    'code': func_entry_node.properties['name'],
                    'type': func_entry_node.properties['type'],
                    'location': func_entry_node.properties['location'],
                    'filepath': filepath,
                    'functionId': str(edge.start_node.properties['functionId'])
                }
            else:
                node_prop = {
                    'code': edge.start_node.properties['code'],
                    'type': edge.start_node.properties['type'],
                    'location': edge.start_node.properties['location'],
                    'filepath': filepath,
                    'functionId': str(edge.start_node.properties['functionId'])
                }
            g.add_vertex(startNode, **node_prop)

        endNode = str(edge.end_node._id)

        if not isNodeExist(g, endNode):
            if edge.end_node.properties['code'] == 'EXIT' and edge.end_node.properties['code'] == 'ERROR':
                continue

            node_prop = {
                'code': edge.end_node.properties['code'],
                'type': edge.end_node.properties['type'],
                'location': edge.end_node.properties['location'],
                'filepath': filepath,
                'functionId': str(edge.end_node.properties['functionId'])
            }
            g.add_vertex(endNode, **node_prop)

        edge_prop = {'var': edge.type}
        g.add_edge(startNode, endNode, **edge_prop)

    return g

def obtainASTByNode(db, func_node):
    func_id = func_node._id
    edges = getDirectChildren(db, func_id)
    g = drawGraph(db, edges, func_node)
    return g

def modifyStmtNode(ast):
    dict_row2nodestmt = {}
    dict_row2nodeid = {}
    dict_static = {}

    i = 0
    while i < ast.vcount():
        if ast.vs[i]['type'] == 'Statement' and ast.vs[i]['code'] == 'static':
            raw = int(ast.vs[i]['location'].split(':')[0])
            col = int(ast.vs[i]['location'].split(':')[1])
            dict_static[raw] = (raw, col)
            ast.delete_vertices(i)
        else:
            i += 1

    i = 0
    while i < ast.vcount():
        if ast.vs[i]['type'] == 'Statement':
            if ast.vs[i]['location'] is None:
                i=i+1
                continue
            row = int(ast.vs[i]['location'].split(':')[0])
            col = int(ast.vs[i]['location'].split(':')[1])
            _tuple = (ast.vs[i]['code'], row, col, ast.vs[i]['location'])

            if row not in dict_row2nodestmt.keys():
                dict_row2nodestmt[row] = [_tuple]
                dict_row2nodeid[row] = ast.vs[i]['name']
                i += 1
            else:
                dict_row2nodestmt[row].append(_tuple)
                ast.delete_vertices(i)
        else:
            i += 1

    j = 0
    list_nodeindex_to_delete = []
    while j < ast.vcount():
        if ast.vs[j]['location'] != None:
            row = int(ast.vs[j]['location'].split(':')[0])
            col = int(ast.vs[j]['location'].split(':')[1])
        else:
            j += 1
            continue

        if row in dict_row2nodestmt.keys() and ast.vs[j]['type'] != 'Statement':
            _tuple = (ast.vs[j]['code'], row, col, ast.vs[j]['location'])
            dict_row2nodestmt[row].append(_tuple)
            list_nodeindex_to_delete.append(ast.vs[j]['name'])
            j += 1
        else:
            j += 1

    for key in dict_row2nodestmt.keys():
        dict_row2nodestmt[key].sort(key=lambda e: e[2])
        nodename = dict_row2nodeid[key]
        nodeIndex = 0
        for node in ast.vs:
            if node['name'] == nodename:
                break
            else:
                nodeIndex += 1

        location = dict_row2nodestmt[key][0][3]
        new_code = ' '.join([_t[0] for _t in dict_row2nodestmt[key]]).strip()

        ast.vs[nodeIndex]['code'] = new_code
        ast.vs[nodeIndex]['location'] = location
        ast.vs[nodeIndex]['type'] = 'Statement'

    for d_name in list_nodeindex_to_delete:
        i = 0
        while i < ast.vcount():
            if ast.vs[i]['name'] == d_name:
                ast.delete_vertices(i)
            else:
                i += 1

    n = 0
    while n < ast.vcount():
        if ast.vs[n]['location'] == None:
            n += 1
            continue

        raw = int(ast.vs[n]['location'].split(':')[0])
        col = int(ast.vs[n]['location'].split(':')[1])
        if raw in dict_static.keys() and col > dict_static[raw][1]:
            ast.vs[n]['code'] = 'static ' + ast.vs[n]['code']

        n += 1

    list_node_index = []
    for node in ast.vs:
        if node['type'] == 'Statement':
            if node['location'] == None:
                continue
            raw = int(node['location'].split(':')[0])
            list_node_index.append((raw, node))

    list_node_index.sort(key=lambda x: (x[0], x[1]))

    i = 1
    list_del_name = []
    while i < len(list_node_index):
        if list_node_index[i][0] - list_node_index[i - 1][0] == 1:
            list_node_index[i][1]['code'] = list_node_index[i - 1][1]['code'] + '\n' + list_node_index[i][1]['code']
            list_del_name.append(list_node_index[i - 1][1]['name'])
            del list_node_index[i - 1]
        else:
            i += 1

    _dict = {}
    for n in list_node_index:
        _dict[n[1]['name']] = n[1]['code']

    j = 0
    while j < ast.vcount():
        if ast.vs[j]['name'] in list_del_name:
            ast.delete_vertices(j)
        elif ast.vs[j]['name'] in _dict.keys():
            ast.vs[j]['code'] = _dict[ast.vs[j]['name']]
            j += 1
        else:
            j += 1

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

def main():
    j = JoernSteps()
    j.connectToDatabase()
    all_func_node = getALLFuncNode(j)

    for node in all_func_node:
        file_path = getFuncFile(j, node._id)
        cweID = file_path.split('/')[-2]
        ast_path = os.path.join("ast", cweID)

        file_name = file_path.split('/')[-1]
        ast_file_name = file_name.split('.')[0] + '_' + str(node._id)
        path = os.path.join(ast_path, ast_file_name)

        if os.path.exists(path):
            continue

        ast_1 = obtainASTByNode(j, node)

        ast_2 = modifyStmtNode(ast_1)

        ast = deleteEdgeOfAST(ast_1)

        if not os.path.exists(ast_path):
            os.mkdir(ast_path)

        with open(path, 'wb') as f:
            pickle.dump(ast, f)

    print("complete!!!")

if __name__ == '__main__':
    main()
