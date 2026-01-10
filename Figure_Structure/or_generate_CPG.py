import shutil
from joern.all import JoernSteps
from igraph import *
import os
import pickle
import re


list_destparam_0_cpyapi = ['sprintf', 'gets', 'fgets', '_memccpy', '_mbscpy', '_strncpy', 'wmemset', 'vasprintf', 'asprintf', 'wcsncpy', 'lstrcpy', '_wcsncpy', '_snprintf', 'memcpy', 'memmove', '_tcscpy', '_snwprintf', 'strcpy', 'CopyMemory', 'wcsncat', 'vsprintf', 'lstrcpyn', 'vsnprintf', '_mbsncat', 'wmemmove', 'memset', 'wmemcpy', 'strcat', 'fprintf', '_strncat', '_tcsncpy', '_mbsnbcpy', 'strncpy', 'strncat', 'wcscpy', 'snprintf', 'lstrcat']
list_scanf_api = ['vfscanf', 'fscanf', 'vscanf', 'scanf', 'vsscanf', 'sscanf', 'swscanf']
list_key_words = []


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
    query_str = """queryNodeIndex('functionId:%s').outE('IS_AST_PARENT')""" % (func_id)
    edges = db.runGremlinQuery(query_str)
    return edges


def getCFGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('FLOWS_TO')""" % (func_id)
    cfgEdges = db.runGremlinQuery(query_str)
    return cfgEdges


def getDDGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('REACHES')""" % (func_id)
    ddgEdges = db.runGremlinQuery(query_str)
    return ddgEdges


def getCDGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('CONTROLS')""" % (func_id)
    cdgEdges = db.runGremlinQuery(query_str)
    return cdgEdges


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


def isDelCPGNode(v):
    if v['code'] == 'EXIT':
        return False
    elif v['location'] == None and v['type'] != 'FunctionDef': 
        return True
    elif v['code'] == '':
        return True
    else:
        return False


def isEdgeExists(cpg, startnode, endnode):
    for edge in cpg.es:
        if cpg.vs[edge.tuple]['name'][0] == startnode and cpg.vs[edge.tuple]['name'][1] == endnode:
            return True
        else:
            continue
    return False


def addCPGEdge(cpg, startnode, endnode):
    if isEdgeExists(cpg, startnode, endnode):
        return cpg

    edge_prop = {'type': 'IS_AST_PARENT', 'var': None}
    cpg.add_edge(startnode, endnode, **edge_prop)
    return cpg


def getUSENodesVar(db, func_id):
    query = "g.v(%s).out('USE').code" % func_id
    ret = db.runGremlinQuery(query)
    if ret == []:
        return False
    else:
        return ret


def getDEFNodesVar(db, func_id):
    query = "g.v(%s).out('DEF').code" % func_id
    ret = db.runGremlinQuery(query)
    if ret == []:
        return False
    else:
        return ret


def getReturnVarOfAPI(code):
    for api in list_destparam_0_cpyapi:
        if code.find(api + ' ') != -1:
            _list = code.split(api + ' ')
            if _list[0] == '' and _list[1][0] == '(':
                var = _list[1].split(',')[0].replace('(', '').strip()
                if var.find(' & ') > -1:
                    var = var.split(' & ')[1]

                if var.find(' + ') != -1:
                    var = var.split(' + ')[0]
                    if var.find(' . ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' . ') != -1:
                            var_1.append(var.split(' . ')[0])
                            _list.append(' . '.join(var_1))
                            var = ' . '.join(var.split(' . ')[1:])

                        return _list

                    elif var.find(' -> ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' -> ') != -1:
                            var_1.append(var.split(' -> ')[0])
                            _list.append(' -> '.join(var_1))
                            var = ' -> '.join(var.split(' -> ')[1:])

                        return _list

                    else:
                        return [var]

                elif var.find(' - ') != -1:
                    var = var.split(' - ')[0]
                    if var.find(' . ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' . ') != -1:
                            var_1.append(var.split(' . ')[0])
                            _list.append(' . '.join(var_1))
                            var = ' . '.join(var.split(' . ')[1:])

                        return _list

                    elif var.find(' -> ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' -> ') != -1:
                            var_1.append(var.split(' -> ')[0])
                            _list.append(' -> '.join(var_1))
                            var = ' -> '.join(var.split(' -> ')[1:])

                        return _list

                    else:
                        return [var]

                elif var.find(' * ') != -1:
                    temp = var.split(' * ')[1]
                    if temp[0] == ')':
                        var = temp[1:].strip()
                    else:
                        var = var.split(' * ')[0]

                    if var.find(' . ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' . ') != -1:
                            var_1.append(var.split(' . ')[0])
                            _list.append(' . '.join(var_1))
                            var = ' . '.join(var.split(' . ')[1:])

                        return _list

                    elif var.find(' -> ') != -1:
                        _list = [var]
                        var_1 = []
                        while var.find(' -> ') != -1:
                            var_1.append(var.split(' -> ')[0])
                            _list.append(' -> '.join(var_1))
                            var = ' -> '.join(var.split(' -> ')[1:])

                        return _list

                    else:
                        return [var]

                elif var.find(' . ') != -1:
                    _list = [var]
                    var_1 = []
                    while var.find(' . ') != -1:
                        var_1.append(var.split(' . ')[0])
                        _list.append(' . '.join(var_1))
                        var = ' . '.join(var.split(' . ')[1:])

                    return _list

                elif var.find(' -> ') != -1:
                    _list = [var]
                    var_1 = []
                    while var.find(' -> ') != -1:
                        var_1.append(var.split(' -> ')[0])
                        _list.append(' -> '.join(var_1))
                        var = ' -> '.join(var.split(' -> ')[1:])

                    return _list

                else:
                    return [var]

        else:
            continue

    for scanfapi in list_scanf_api:
        if scanfapi in ['fscanf', 'sscanf', 'swscanf', 'vfscanf', 'vsscanf']:
            if code.find(scanfapi + ' ') != -1:
                _list = code.split(scanfapi+' ')
                if _list[0] == '' and _list[1][0] == '(':
                    list_var = _list[1].split(',')[2:]
                    list_var = [var.replace('(', '').strip() for var in list_var]
                    new_list_var = []
                    for var in list_var:
                        if var.find(' & ') > -1:
                            var = var.split(' & ')[1]

                        if var.find(' + ') > -1:
                            var = var.split(' + ')[0]
                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            elif var.find(' -> ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' -> ') != -1:
                                    var_1.append(var.split(' -> ')[0])
                                    _list.append(' -> '.join(var_1))
                                    var = ' -> '.join(var.split(' -> ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' - ') != -1:
                            var = var.split(' - ')[0]
                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            elif var.find(' -> ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' -> ') != -1:
                                    var_1.append(var.split(' -> ')[0])
                                    _list.append(' -> '.join(var_1))
                                    var = ' -> '.join(var.split(' -> ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' * ') != -1:
                            temp = var.split(' * ')[1]
                            if temp[0] == ')':
                                var = temp[1:].strip()
                            else:
                                var = var.split(' * ')[0]

                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            elif var.find(' -> ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' -> ') != -1:
                                    var_1.append(var.split(' -> ')[0])
                                    _list.append(' -> '.join(var_1))
                                    var = ' -> '.join(var.split(' -> ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' . ') != -1:
                            _list = [var]
                            var_1 = []
                            while var.find(' . ') != -1:
                                var_1.append(var.split(' . ')[0])
                                _list.append(' . '.join(var_1))
                                var = ' . '.join(var.split(' . ')[1:])

                            new_list_var += _list

                        elif var.find(' -> ') != -1:
                            _list = [var]
                            var_1 = []
                            while var.find(' -> ') != -1:
                                var_1.append(var.split(' -> ')[0])
                                _list.append(' -> '.join(var_1))
                                var = ' -> '.join(var.split(' -> ')[1:])

                            new_list_var += _list

                        else:
                            new_list_var.append(var)

                    return new_list_var


        elif scanfapi in ['scanf', 'vscanf']:
            if code.find(scanfapi) != -1:
                _list = code.split(scanfapi + ' ')
                if _list[0] == '' and _list[1][0] == '(':
                    list_var = _list[1].split(',')[1:]
                    list_var = [var.replace('(', '').strip() for var in list_var]
                    new_list_var = []
                    for var in list_var:
                        if var.find(' & ') > -1:
                            var = var.split(' & ')[1]

                        if var.find(' + ') != -1:
                            var = var.split(' + ')[0]
                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' - ') != -1:
                            var = var.split(' - ')[0]
                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' * ') != -1:
                            temp = var.split(' * ')[1]
                            if temp[0] == ')':
                                var = temp[1:].strip()
                            else:
                                var = var.split(' * ')[0]

                            if var.find(' . ') != -1:
                                _list = [var]
                                var_1 = []
                                while var.find(' . ') != -1:
                                    var_1.append(var.split(' . ')[0])
                                    _list.append(' . '.join(var_1))
                                    var = ' . '.join(var.split(' . ')[1:])

                                new_list_var += _list

                            else:
                                new_list_var.append(var)

                        elif var.find(' . ') != -1:
                            _list = [var]
                            var_1 = []
                            while var.find(' . ') != -1:
                                var_1.append(var.split(' . ')[0])
                                _list.append(' . '.join(var_1))
                                var = ' . '.join(var.split(' . ')[1:])

                            new_list_var += _list

                        elif var.find(' -> ') != -1:
                            _list = [var]
                            var_1 = []
                            while var.find(' -> ') != -1:
                                var_1.append(var.split(' -> ')[0])
                                _list.append(' -> '.join(var_1))
                                var = ' -> '.join(var.split(' -> ')[1:])

                            new_list_var += _list

                        else:
                            new_list_var.append(var)

                    return new_list_var

    return False


def getVarOfNode(code):
    list_var = []
    if code.find(' = ') != -1:
        _list = code.split(' = ')[0].split(' ')
        if ']' in _list:
            index_right = _list.index(']')
            index_left = _list.index('[')
            i = 0
            while i < len(_list):
                if i < index_left or i > index_right:
                    list_var.append(_list[i])
                i += 1

    elif code[-1] == ';':
        code = code[:-1].strip()
        if '(' in code:
            list_var = False
        else:
            list_value = code.split(',')
            for _list in list_value:
                _list = code.split(' ')
                if '[' in _list:
                    index = _list.index('[')
                    var = _list[index - 1]
                    list_var.append(var)
                else:
                    var = _list[-1]
                    list_var.append(var)
    else:
        if '(' in code:
            list_var = False
        else:
            list_value = code.split(',')
            for _list in list_value:
                _list = code.split(' ')
                if '[' in _list:
                    index = _list.index('[')
                    var = _list[index - 1]
                    list_var.append(var)
                else:
                    var = _list[-1]
                    list_var.append(var)

    return list_var


def isDataEdgeExist(cpg, startnode, endnode, var):
    for edge in cpg.es:
        if cpg.vs[edge.tuple]['name'][0] == startnode and cpg.vs[edge.tuple]['name'][1] == endnode and edge['var'] == var:
            return True
        else:
            continue
    return False


def addDataEdge(cpg, startnode, endnode, var): 
    if isDataEdgeExist(cpg, startnode, endnode, var):
        return cpg

    edge_prop = {'type': 'REACHES', 'var': var}
    cpg.add_edge(startnode, endnode, **edge_prop)
    return cpg


def get_ifname(node_id, dict_if2cfgnode, dict_cfgnode2if):
    if_name = ''
    min_count = 10000000
    for if_n in dict_cfgnode2if[node_id]:
        if len(dict_if2cfgnode[if_n]) < min_count:
            min_count = len(dict_if2cfgnode[if_n])
            if_name = if_n
        else:
            continue

    return if_name



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
            if (edge.type == 'REACHES' or edge.type == 'CONTROLS') and edge.end_node.properties['code'] == 'EXIT':
                continue
            if edge.end_node.properties['code'] == 'ERROR':
                continue

            node_prop = {
                'code': edge.end_node.properties['code'],
                'type': edge.end_node.properties['type'],
                'location': edge.end_node.properties['location'],
                'filepath': filepath,
                'functionId': str(edge.end_node.properties['functionId'])
            }
            g.add_vertex(endNode, **node_prop)

        if edge.type == 'FLOWS_TO':
            edge_prop = {'type': edge.type, 'var': edge.properties['flowLabel']}
        else:
            edge_prop = {'type': edge.type, 'var': edge.properties['var']}

        g.add_edge(startNode, endNode, **edge_prop)

    return g



def obtainCPGByNode(db, func_node):
    func_id = func_node._id
    ast_edges = getDirectChildren(db, func_id) 
    cfg_edges = getCFGEdges(db, func_id) 
    ddg_Edges = getDDGEdges(db, func_id)  
    cdg_Edges = getCDGEdges(db, func_id)  

    Edges = cfg_edges + ddg_Edges + cdg_Edges + ast_edges  

    g = drawGraph(db, Edges, func_node)
    return g



def deleteNodeOfCPG(cpg):
    vertices_to_delete = []
    for vertex in cpg.vs:
        if isDelCPGNode(vertex): 
            list_pre = vertex.predecessors() 
            list_su = vertex.successors() 

            if len(list_su) != 0:
                for i in range(len(list_pre)):
                    for j in range(len(list_su)):
                        addCPGEdge(cpg, list_pre[i], list_su[j])

            vertices_to_delete.append(vertex.index)

    cpg.delete_vertices(vertices_to_delete)
    return cpg


def deleteEdgeOfCPG(cpg):
    edge_to_delete = []
    all_edges = []
    for edge in cpg.es:
        source_node = edge.source
        target_node = edge.target
        edge_tuple = (source_node, target_node)
        if edge_tuple in all_edges:
            edge_to_delete.append(edge.index)
        else:
            all_edges.append(edge_tuple)

    cpg.delete_edges(edge_to_delete)
    return cpg



def getSubGraph(startNode, list_node, not_scan_list):
    if startNode['name'] in not_scan_list or startNode['code'] == 'EXIT':
        return list_node, not_scan_list
    else:
        list_node.append(startNode)
        not_scan_list.append(startNode['name']) 

    successors = startNode.successors()
    if successors != []:
        for p_node in successors:
            list_node, not_scan_list = getSubGraph(p_node, list_node, not_scan_list)

    return list_node, not_scan_list


def getCtrlRealtionOfCPG(cpg):
    list_ifstmt_nodes = [] 

    for node in cpg.vs:
        if node['type'] == 'Condition': 
            filepath = node['filepath']
            location_row = int(node['location'].split(':')[0])  
            fin = open(filepath, 'r')
            content = fin.readlines()  
            fin.close()
            src_code = content[location_row-1]

            pattern = re.compile("(?:if|while|for|switch)")
            result = re.search(pattern, src_code)  

            if result == None:
                res = 'for'
            else:
                res = result.group()

            if res == '':
                print "error!"
            elif res == 'if':
                list_ifstmt_nodes.append(node)
            else:
                continue
        else:
            continue

    _dict = {} 
    for if_node in list_ifstmt_nodes: 
        list_truestmt_nodes = [] 
        list_falsestmt_nodes = [] 

        for es in cpg.es:
            if cpg.vs[es.tuple[0]] == if_node and es['var'] == 'True':
                start_node = cpg.vs[es.tuple[1]]  
                not_scan_list = [if_node['name']] 

                list_truestmt_nodes, temp = getSubGraph(start_node, list_truestmt_nodes, not_scan_list)
            elif cpg.vs[es.tuple[0]] == if_node and es['var'] == 'False':
                start_node = cpg.vs[es.tuple[1]]
                not_scan_list = [if_node['name']]
                list_falsestmt_nodes, temp = getSubGraph(start_node, list_falsestmt_nodes, not_scan_list)
            else:
                continue

        _share_list = []
        for t_node in list_truestmt_nodes:
            if t_node in list_falsestmt_nodes:
                _share_list.append(t_node)
            else:
                continue
        if _share_list != []:
            i = 0
            while i < len(list_truestmt_nodes):
                if list_truestmt_nodes[i] in _share_list:
                    del list_truestmt_nodes[i]
                else:
                    i += 1
            i = 0
            while i < len(list_falsestmt_nodes):
                if list_falsestmt_nodes[i] in _share_list:
                    del list_falsestmt_nodes[i]
                else:
                    i += 1

            _dict[if_node['name']] = ([t_node['name'] for t_node in list_truestmt_nodes], [f_node['name'] for f_node in list_falsestmt_nodes])
        else:
            filepath = cpg.vs[0]['filepath']
            fin = open(filepath, 'r')
            content = fin.readlines()
            fin.close()

            if_line = int(if_node['location'].split(':')[0])-1 

            if list_truestmt_nodes == []: 
                continue
            sorted_list_truestmt_nodes = sortedNodesByLoc(list_truestmt_nodes) 

            true_stmt_start = sorted_list_truestmt_nodes[0] 
            start_line = int(true_stmt_start['location'].split(':')[0]) 
            str_if_stmts = '\n'.join(content[if_line:start_line])

            if '{' in str_if_stmts:
                if sorted_list_truestmt_nodes[-1]['location'] == None:
                    end_line = int(sorted_list_truestmt_nodes[-2]['location'].split(':')[0])
                else:
                    end_line = int(sorted_list_truestmt_nodes[-1]['location'].split(':')[0])

                list_stmt = content[if_line:end_line]
                left_brace = 0
                i = 0
                index = 0
                tag = 0
                for stmt in list_stmt: 
                    for c in stmt:
                        if c == '{': 
                            left_brace += 1

                        elif c == '}':
                            left_brace -= 1

                            if left_brace == 0: 
                                tag = 1
                                break

                    if tag == 1:
                        break
                    else:
                        index += 1 
                real_end_line = int(if_node['location'].split(':')[0]) + index 
                list_real_true_stmt = []  

                for node in sorted_list_truestmt_nodes: 
                    if node['location'] == None:
                        continue

                    if int(node['location'].split(':')[0]) >= if_line+1 and int(node['location'].split(':')[0]) <= real_end_line: 
                        list_real_true_stmt.append(node) 
            else:
                list_real_true_stmt = [true_stmt_start] 

            if list_falsestmt_nodes == []:
                continue
            sorted_list_falsestmt_nodes = sortedNodesByLoc(list_falsestmt_nodes) 

            false_stmt_start = sorted_list_falsestmt_nodes[0] 
            if sorted_list_truestmt_nodes[-1]['location'] != None:
                start_line = int(sorted_list_truestmt_nodes[-1]['location'].split(':')[0])
            else:
                start_line = int(sorted_list_truestmt_nodes[-2]['location'].split(':')[0])
            end_line = int(false_stmt_start['location'].split(':')[0]) 

            str_else_stmts = '\n'.join(content[start_line:end_line])
            if 'else' in str_else_stmts:
                else_line = 0 
                for line in content[start_line:end_line]: 
                    if 'else' in line: 
                        break
                    else:
                        else_line += 1

                real_else_line = start_line + else_line + 1 
                str_else_stmts = str_else_stmts.split('else')[1] 

                if '{' in str_else_stmts:
                    if sorted_list_falsestmt_nodes[-1]['location'] != None:
                        end_line = int(sorted_list_falsestmt_nodes[-1]['location'].split(':')[0])
                    elif sorted_list_falsestmt_nodes[-2]['location'] != None:
                        end_line = int(sorted_list_falsestmt_nodes[-2]['location'].split(':')[0])
                    else:
                        end_line = int(sorted_list_falsestmt_nodes[-3]['location'].split(':')[0])
                    list_stmt = content[real_else_line-1:end_line]
                    left_brace = 0
                    i = 0
                    index = 0
                    tag = 0
                    for stmt in list_stmt:
                        for c in stmt:
                            if c == '{':
                                left_brace += 1

                            elif c == '}':
                                left_brace -= 1

                                if left_brace == 0:
                                    tag = 1
                                    break

                        if tag == 1:
                            break
                        else:
                            index += 1

                    real_end_line = int(if_node['location'].split(':')[0]) + index
                    list_real_false_stmt = []

                    for node in sorted_list_falsestmt_nodes:
                        if node['location'] == None:
                            continue
                        if int(node['location'].split(':')[0]) >= if_line+1 and int(node['location'].split(':')[0]) <= real_end_line:
                            list_real_false_stmt.append(node)
                else:
                    list_real_false_stmt = [false_stmt_start]
            else:
                list_real_false_stmt = []

            _dict[if_node['name']] = ([t_node['name'] for t_node in list_real_true_stmt], [f_node['name'] for f_node in list_real_false_stmt])
    return _dict


def getUseDefVarByCPG(db, cpg):
    dict_cfg2use = {} 
    dict_cfg2def = {} 

    for node in cpg.vs: 
        if node['type'] == 'Function':
            continue

        func_id = node['name'] 
        use_node = getUSENodesVar(db, func_id) 
        def_node = getDEFNodesVar(db, func_id)

        if node['type'] == 'Statement':
            if def_node == False:
                code = node['code'].replace('\n', ' ') 
                if code.find(" = ") != -1:
                    value = code.split(" = ")[0].strip().split(' ')
                    if value[-1] == ']':
                        newvalue = code.split(" [ ")[0].strip().split(' ') 
                        if '->' in newvalue:
                            a_index = newvalue.index('->')  
                            n_value = ' '.join(
                                [newvalue[a_index - 1], '->', newvalue[a_index + 1]])  
                            newvalue[a_index - 1] = n_value 
                            del newvalue[a_index]
                            del newvalue[a_index]

                        def_node = newvalue
                    else:  
                        if '->' in value:
                            a_index = value.index('->')  
                            n_value = ' '.join([value[a_index - 1], '->', value[a_index + 1]]) 
                            ob_value = value[a_index - 1] 
                            value[a_index - 1] = n_value 
                            del value[a_index]
                            del value[a_index]
                            value.append(ob_value.replace('*', '')) 

                        def_node = value

            if use_node == False:
                code = node['code'].replace('\n', ' ')  
                if code.find(" = ") != -1:
                    value = code.split(" = ")[1].strip().split(' ') 
                    newvalue = []
                    for v in value: 
                        if v == '*' or v == '+' or v == '-' or v == '->' or v == '(' or v == ')' or v == '[' or v == ']' or v == '&' or v == '.' or v == '::' or v == ';' or v == ',':
                            continue
                        else:
                            newvalue.append(v.strip())
                else:
                    value = code.split(' ')
                    newvalue = []
                    for v in value:
                        if v == '*' or v == '+' or v == '-' or v == '->' or v == '(' or v == ')' or v == '[' or v == ']' or v == '&' or v == '.' or v == '::' or v == ';' or v == ',':
                            continue
                        else:
                            newvalue.append(v.strip())
                use_node = newvalue

        if use_node:
            use_node = [code.replace('*', '').replace('&', '').strip() for code in use_node] 

        if def_node:
            def_node = [code.replace('*', '').replace('&', '').strip() for code in def_node]
        else:
            new_def_node = getReturnVarOfAPI(node['code'])  
            if node['name'] == '2078': 
                print("new_def_node", new_def_node)

            if new_def_node:
                def_node = []
                for code in new_def_node:
                    new_code = code.replace('*', '').replace('&', '').strip()
                    def_node.append(new_code)

                    if new_code not in use_node: 
                        use_node.append(new_code) 

        if use_node:
            dict_cfg2use[node['name']] = use_node 

        if def_node: 
            dict_cfg2def[node['name']] = def_node  

    return dict_cfg2use, dict_cfg2def


def modifyDataEdgeVal(cpg):
    for edge in cpg.es:
        if edge['var'] == None: 
            continue

        new_val = ''
        for c in edge['var']: 
            if c == '*': 
                continue
            else:
                new_val += c 

        edge['var'] = new_val  

    return cpg


def getInitNodeOfDecl(cpg, list_sorted_pdgnode, node, var, dict_use, dict_def):
    index = list_sorted_pdgnode.index(node) 
    list_init_node = [] 

    for i in range(index + 1, len(list_sorted_pdgnode)):
        if list_sorted_pdgnode[i]['type'] != 'IdentifierDeclStatement' and list_sorted_pdgnode[i]['name'] in dict_def.keys():
            if var in dict_def[list_sorted_pdgnode[i]['name']]:
                if isDataEdgeExist(cpg, node['name'], list_sorted_pdgnode[i]['name'], var):
                    continue
                else:
                    list_init_node.append((list_sorted_pdgnode[i], i)) 
        elif list_sorted_pdgnode[i]['type'] != 'IdentifierDeclStatement' and list_sorted_pdgnode[i][
            'name'] not in dict_def.keys():
            if list_sorted_pdgnode[i]['name'] in dict_use.keys() and var in dict_use[list_sorted_pdgnode[i]['name']]:
                if isDataEdgeExist(cpg, node['name'], list_sorted_pdgnode[i]['name'], var):
                    continue
                else:
                    list_init_node.append((list_sorted_pdgnode[i], i))
        else:
            continue 

    return list_init_node  


def completeDeclStmtOfCPG(cpg, dict_use, dict_def, dict_if2cfgnode, dict_cfgnode2if):
    list_sorted_pdgnode = sortedNodesByLoc(cpg.vs)  
    dict_declnode2val = {} 

    for node in cpg.vs:
        if (node['type'] == 'IdentifierDeclStatement' or node['type'] == 'Parameter' or node['type'] == 'Statement') and node['code'].find(' = ') == -1:
            if node['type'] == 'IdentifierDeclStatement' or node['type'] == 'Parameter': 
                list_var = dict_def[node['name']] 
            else:
                list_var = getVarOfNode(node['code'])  

            if list_var == False: 
                continue
            else:
                for var in list_var: 
                    results = getInitNodeOfDecl(cpg, list_sorted_pdgnode, node, var, dict_use, dict_def)
                    if results != []:  
                        for result in results:
                           
                            if node['name'] not in dict_cfgnode2if.keys():
                                startnode = node['name']
                                endnode = result[0]['name'] 
                                cpg = addDataEdge(cpg, startnode, endnode, var)  
                            else:
                                list_if = dict_cfgnode2if[node['name']] 
                                list_not_scan = []  

                                for ifstmt_n in list_if:
                                    tuple_statements = dict_if2cfgnode[ifstmt_n] 
                                    if node['name'] in tuple_statements[0]: 
                                        list_not_scan += tuple_statements[1] 
                                    elif node['name'] in tuple_statements[1]: 
                                        list_not_scan += tuple_statements[0] 

                                if result[0]['name'] not in list_not_scan: 
                                    startnode = node['name'] 
                                    endnode = result[0]['name']  
                                    cpg = addDataEdge(cpg, startnode, endnode, var) 

    return cpg


def get_nodes_before_exit(cpg, dict_if2cfgnode, dict_cfgnode2if):
    _dict = {} 
    for key in dict_cfgnode2if.keys(): 
        results = cpg.vs.select(name=key) 
        if len(results) != 0 and (results[0]['type'] == 'BreakStatement' or results[0]['type'] == 'ReturnStatement' or results[0]['code'].find('exit ') != -1 or results[0]['type'] == 'GotoStatement'):
            if len(dict_cfgnode2if[key]) == 1: 
                if_name = dict_cfgnode2if[key][0]  
            else:
                if_name = get_ifname(key, dict_if2cfgnode, dict_cfgnode2if) 
            _list_name_0 = dict_if2cfgnode[if_name][0]  
            _list_name_1 = dict_if2cfgnode[if_name][1] 

            if key in _list_name_0: 
                ret_index = _list_name_0.index(key) 
                del _list_name_0[ret_index] 
                for name in _list_name_0: 
                    _dict[name] = key 

            if key in _list_name_1:
                ret_index = _list_name_1.index(key)
                del _list_name_1[ret_index]
                for name in _list_name_1:
                    _dict[name] = key

        else:
            continue  

    return _dict


def completeDataEdgeOfCPG(cpg, dict_use, dict_def, dict_if2cfgnode, dict_cfgnode2if):
    list_sorted_pdgnode = sortedNodesByLoc(cpg.vs) 
    exit2stmt_dict = get_nodes_before_exit(cpg, dict_if2cfgnode, dict_cfgnode2if) 

    for i in range(0, len(list_sorted_pdgnode)):
        if list_sorted_pdgnode[i]['type'] == 'IdentifierDeclStatement':
            continue 
        if list_sorted_pdgnode[i]['name'] in dict_def.keys():
            list_def_var = dict_def[list_sorted_pdgnode[i]['name']]  

            for def_var in list_def_var: 
                for j in range(i+1, len(list_sorted_pdgnode)):
                    if list_sorted_pdgnode[i]['name'] in exit2stmt_dict.keys():  
                        exit_name = exit2stmt_dict[list_sorted_pdgnode[i]['name']]  

                        if list_sorted_pdgnode[j]['name'] == exit_name: 
                            break
                        elif list_sorted_pdgnode[j]['name'] in dict_use.keys() and def_var in dict_use[list_sorted_pdgnode[j]['name']]:
                            if list_sorted_pdgnode[i]['name'] not in dict_cfgnode2if.keys():
                                startnode = list_sorted_pdgnode[i]['name']
                                endnode = list_sorted_pdgnode[j]['name']
                                addDataEdge(cpg, startnode, endnode, def_var)
                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
                            elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2if.keys() and list_sorted_pdgnode[j]['name'] not in dict_cfgnode2if.keys():
                                startnode = list_sorted_pdgnode[i]['name']
                                endnode = list_sorted_pdgnode[j]['name']
                                addDataEdge(cpg, startnode, endnode, def_var)

                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
                            elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2if.keys() and list_sorted_pdgnode[j]['name'] in dict_cfgnode2if.keys():
                                if_list = dict_cfgnode2if[list_sorted_pdgnode[i]['name']] 
                                _not_scan = []  
                                for if_stmt in if_list: 
                                    _tuple = dict_if2cfgnode[if_stmt]  
                                    if list_sorted_pdgnode[i]['name'] in _tuple[0]:  
                                        _not_scan += _tuple[1] 
                                    else:
                                        _not_scan += _tuple[0]

                                if list_sorted_pdgnode[j]['name'] not in _not_scan:  
                                    startnode = list_sorted_pdgnode[i]['name']
                                    endnode = list_sorted_pdgnode[j]['name']
                                    addDataEdge(cpg, startnode, endnode, def_var) 

                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
                    else: 
                        if list_sorted_pdgnode[j]['name'] in dict_use.keys() and def_var in dict_use[list_sorted_pdgnode[j]['name']]:
                            if list_sorted_pdgnode[i]['name'] not in dict_cfgnode2if.keys():
                                startnode = list_sorted_pdgnode[i]['name']
                                endnode = list_sorted_pdgnode[j]['name']
                                addDataEdge(cpg, startnode, endnode, def_var)
                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
                            elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2if.keys() and list_sorted_pdgnode[j]['name'] not in dict_cfgnode2if.keys():
                                startnode = list_sorted_pdgnode[i]['name']
                                endnode = list_sorted_pdgnode[j]['name']
                                addDataEdge(cpg, startnode, endnode, def_var)
                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
                            elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2if.keys() and list_sorted_pdgnode[j]['name'] in dict_cfgnode2if.keys():
                                if_list = dict_cfgnode2if[list_sorted_pdgnode[i]['name']]
                                _not_scan = []
                                for if_stmt in if_list:
                                    _tuple = dict_if2cfgnode[if_stmt]
                                    if list_sorted_pdgnode[i]['name'] in _tuple[0]:
                                        _not_scan += _tuple[1]
                                    else:
                                        _not_scan += _tuple[0]
                                if list_sorted_pdgnode[j]['name'] not in _not_scan:
                                    startnode = list_sorted_pdgnode[i]['name']
                                    endnode = list_sorted_pdgnode[j]['name']
                                    addDataEdge(cpg, startnode, endnode, def_var)
                                if list_sorted_pdgnode[j]['name'] in dict_def.keys() and def_var in dict_def[list_sorted_pdgnode[j]['name']]:
                                    break
        else:
            continue

    return cpg


def addDataEdgeOfObject(cpg, dict_if2cfgnode, dict_cfgnode2if):
    for node in cpg.vs:
        if node['code'].find(' = new ') != -1:  
            objectname = node['code'].split(' = new ')[0].split(' ')[-1].strip() 
            cur_name = node['name'] 

            for pnode in cpg.vs: 
                if pnode['name'] == cur_name: 
                    continue
                if node['name'] not in dict_cfgnode2if.keys():
                    if pnode['code'].find(objectname + ' -> ') != -1:
                        if pnode['code'].split(objectname + ' -> ')[0] == '':
                            startnode = node['name']
                            endnode = pnode['name']
                            def_var = objectname
                            addDataEdge(cpg, startnode, endnode, def_var) 
                        elif pnode['code'].split(objectname + ' -> ')[0][-1] == ' ':
                            startnode = node['name']
                            endnode = pnode['name']
                            def_var = objectname
                            addDataEdge(cpg, startnode, endnode, def_var)
                    elif pnode['code'].find('delete ') != -1: 
                        startnode = node['name']
                        endnode = pnode['name']
                        def_var = objectname
                        addDataEdge(cpg, startnode, endnode, def_var)
                    else:
                        continue
                else:
                    list_if = dict_cfgnode2if[node['name']]  
                    list_not_scan = []  

                    for ifstmt_n in list_if:
                        tuple_statements = dict_if2cfgnode[ifstmt_n] 
                        if node['name'] in tuple_statements[0]:  
                            list_not_scan += tuple_statements[1] 
                        elif node['name'] in tuple_statements[1]:
                            list_not_scan += tuple_statements[0]
                    if pnode['code'].find(objectname + ' -> ') != -1 and pnode['name'] not in list_not_scan:
                        if pnode['code'].split(objectname + ' -> ')[0] == '':
                            startnode = node['name']
                            endnode = pnode['name']
                            def_var = objectname
                            addDataEdge(cpg, startnode, endnode, def_var)
                        elif pnode['code'].split(objectname + ' -> ')[0][-1] == ' ' :
                            startnode = node['name']
                            endnode = pnode['name']
                            def_var = objectname
                            addDataEdge(cpg, startnode, endnode, def_var)
                    elif pnode['code'].find('delete ') != -1  and pnode['name'] not in list_not_scan:
                        startnode = node['name']
                        endnode = pnode['name']
                        def_var = objectname
                        addDataEdge(cpg, startnode, endnode, def_var)
                    else:
                        continue
        else:
            continue

    return cpg


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


def main():
    j = JoernSteps()
    j.connectToDatabase()
    all_func_node = getALLFuncNode(j)  

    for node in all_func_node:
        file_path = getFuncFile(j, node._id)
        cweID = file_path.split('/')[-2] 
        cpg_path = os.path.join("cpg", cweID)

        file_name = file_path.split('/')[-1]
        cpg_file_name = file_name.split('.')[0] + '_' + str(node._id)
        path = os.path.join(cpg_path, cpg_file_name)

        if os.path.exists(path):
            continue

        cpg_1 = obtainCPGByNode(j, node) 
        cpg_2 = deleteNodeOfCPG(cpg_1)
        cpg_2 = deleteEdgeOfCPG(cpg_2)  

        dict_if2cfgnode = getCtrlRealtionOfCPG(cpg_2)
        dict_cfgnode2if = {} 
        for key in dict_if2cfgnode.keys(): 
            _list = dict_if2cfgnode[key][0] + dict_if2cfgnode[key][1]  
            for v in _list:
                if v not in dict_cfgnode2if.keys():  
                    dict_cfgnode2if[v] = [key]
                else:
                    dict_cfgnode2if[v].append(key)

        for key in dict_cfgnode2if.keys(): 
            dict_cfgnode2if[key] = list(set(dict_cfgnode2if[key]))

        d_use, d_def = getUseDefVarByCPG(j, cpg_2) 

        cpg_3 = modifyDataEdgeVal(cpg_2)
        cpg_4 = completeDeclStmtOfCPG(cpg_3, d_use, d_def, dict_if2cfgnode, dict_cfgnode2if)
        cpg_5 = completeDataEdgeOfCPG(cpg_4, d_use, d_def, dict_if2cfgnode, dict_cfgnode2if)
        cpg_6 = addDataEdgeOfObject(cpg_5, dict_if2cfgnode, dict_cfgnode2if) 

        with open(path, 'wb') as f:
            pickle.dump(cpg_6, f)  
    

if __name__ == '__main__':
    main()
