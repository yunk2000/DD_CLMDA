import javalang
import os
import re
import pickle
import json
from collections import defaultdict
from igraph import Graph, plot
import traceback

node_id_counter = 0
method_nodes = defaultdict(list)
variable_definitions = {}
method_definitions = {}
class_definitions = {}

REQUIRED_VERTEX_ATTRIBUTES = [
    "type", "location", "code", "filepath", "method", "class_name"
]

REQUIRED_EDGE_ATTRIBUTES = [
    "type", "curved"
]


def safe_encode(text):
    if not isinstance(text, str):
        return str(text)

    try:
        return text.encode('utf-8', 'ignore').decode('utf-8')
    except UnicodeEncodeError:
        return text.encode('utf-8', 'replace').decode('utf-8')
    except:
        return text[:100] + "..." if len(text) > 100 else text


class CPGNode:
    def __init__(self, node_id, node_type, location, code, filepath, method=None, class_name=None):
        self.id = node_id
        self.type = node_type
        self.location = location
        self.code = safe_encode(code) 
        self.filepath = filepath
        self.method = method
        self.class_name = class_name
        self.vars_defined = set()
        self.vars_used = set()
        self.calls = []  

    def __repr__(self):
        return f"CPGNode({self.id}, {self.type}, L{self.location}: {self.code})"


def generate_node_id():
    global node_id_counter
    node_id_counter += 1
    return node_id_counter


def reset_global_state():
    global node_id_counter, method_nodes, variable_definitions, method_definitions, class_definitions
    node_id_counter = 0
    method_nodes.clear()
    variable_definitions.clear()
    method_definitions.clear()
    class_definitions.clear()


def extract_variables_from_code(code):
    if not isinstance(code, str):
        return set()
    return set(re.findall(r'\b[a-zA-Z_]\w*\b', code))


def handle_value(value):
    if isinstance(value, (str, int, float, bool)):
        return safe_encode(str(value)) 
    return safe_encode(str(value))


def is_control_flow_node(node):
    control_flow_types = [
        'IfStatement', 'WhileStatement', 'ForStatement',
        'DoStatement', 'SwitchStatement', 'TryStatement',
        'CatchClause', 'ReturnStatement', 'BreakStatement',
        'ContinueStatement', 'ThrowStatement'
    ]
    return type(node).__name__ in control_flow_types


def get_method_args(method_invocation):
    args = []
    if hasattr(method_invocation, 'arguments') and method_invocation.arguments:
        for arg in method_invocation.arguments:
            if hasattr(arg, 'value'):
                args.append(safe_encode(str(arg.value))) 
            else:
                args.append(safe_encode(str(arg)))
    return args


def build_cpg_from_ast(tree, file_path):
    reset_global_state()
    cpg = Graph(directed=True)
    nodes = {}
    edges = []
    entry_node_id = generate_node_id()
    entry_node = CPGNode(entry_node_id, "ENTRY", 0, "ENTRY", file_path)
    nodes[entry_node_id] = entry_node
    cpg.add_vertex(str(entry_node_id))
    for attr in REQUIRED_VERTEX_ATTRIBUTES:
        if attr not in cpg.vertex_attributes():
            cpg.vs[attr] = [None] * len(cpg.vs)
    v = cpg.vs.find(str(entry_node_id))
    v["type"] = entry_node.type
    v["location"] = entry_node.location
    v["code"] = entry_node.code
    v["filepath"] = entry_node.filepath
    v["method"] = entry_node.method
    v["class_name"] = entry_node.class_name
    current_class = None
    current_method = None
    def traverse(node, parent_id=None):
        nonlocal current_class, current_method
        node_id = generate_node_id()
        node_type = type(node).__name__
        location = node.position.line if hasattr(node, 'position') and node.position else None
        code = handle_value(getattr(node, 'value', str(node)))
        cpg_node = CPGNode(node_id, node_type, location, code, file_path, current_method, current_class)
        nodes[node_id] = cpg_node
        cpg.add_vertex(str(node_id))
        if len(cpg.vs) > 0:
            for attr in REQUIRED_VERTEX_ATTRIBUTES:
                if attr not in cpg.vertex_attributes():
                    cpg.vs[attr] = [None] * len(cpg.vs)
        v = cpg.vs.find(str(node_id))
        v["type"] = cpg_node.type
        v["location"] = cpg_node.location
        v["code"] = cpg_node.code
        v["filepath"] = cpg_node.filepath
        v["method"] = cpg_node.method
        v["class_name"] = cpg_node.class_name
        if current_method:
            method_nodes[current_method].append(node_id)
        if node_type == 'ClassDeclaration':
            current_class = node.name
            cpg_node.class_name = current_class
            v["class_name"] = current_class
            class_definitions[current_class] = node_id
        if node_type == 'MethodDeclaration':
            current_method = node.name
            cpg_node.method = current_method
            v["method"] = current_method
            method_definitions[current_method] = node_id
            edges.append((entry_node_id, node_id, "ENTRY"))
        if node_type == 'VariableDeclaration' and hasattr(node, 'declarators'):
            for declarator in node.declarators:
                var_name = declarator.name
                cpg_node.vars_defined.add(var_name)
                variable_definitions[var_name] = node_id
        elif node_type == 'Assignment':
            if hasattr(node, 'expressionl') and hasattr(node.expressionl, 'member'):
                var_name = node.expressionl.member
                cpg_node.vars_defined.add(var_name)
                variable_definitions[var_name] = node_id
        elif node_type == 'MethodInvocation' and hasattr(node, 'member'):
            method_name = node.member
            args = get_method_args(node)
            cpg_node.calls.append((method_name, args))
        variables = extract_variables_from_code(code)
        for var in variables:
            if var not in ['true', 'false', 'null', 'this', 'super']:
                cpg_node.vars_used.add(var)
        if parent_id is not None:
            edges.append((parent_id, node_id, "AST"))
        if parent_id is not None and is_control_flow_node(node):
            parent_node = nodes.get(parent_id)
            if parent_node and is_control_flow_node(parent_node):
                edges.append((parent_id, node_id, "CONTROL"))
        if isinstance(node, javalang.ast.Node):
            for field in node.attrs:
                value = getattr(node, field)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, javalang.ast.Node):
                            traverse(item, node_id)
                elif isinstance(value, javalang.ast.Node):
                    traverse(value, node_id)
    try:
        traverse(tree)
    except Exception as e:
        traceback.print_exc()
        return None
    for node_id, node in nodes.items():
        for var in node.vars_used:
            if var in variable_definitions:
                def_node_id = variable_definitions[var]
                if str(def_node_id) != str(node_id):
                    edges.append((def_node_id, node_id, "DATA"))
    for node_id, node in nodes.items():
        for call in node.calls:
            method_name, args = call
            if method_name in method_definitions:
                called_method_id = method_definitions[method_name]
                edges.append((node_id, called_method_id, "CALL"))
    for method, node_ids in method_nodes.items():
        if len(node_ids) > 1:
            sorted_nodes = sorted(node_ids, key=lambda nid: nodes[nid].location or 0)
            for i in range(len(sorted_nodes) - 1):
                src = sorted_nodes[i]
                tgt = sorted_nodes[i + 1]
                if nodes[src].location and nodes[tgt].location and nodes[src].location < nodes[tgt].location:
                    edges.append((src, tgt, "FLOW"))
    for src, tgt, edge_type in edges:
        src_str = str(src)
        tgt_str = str(tgt)
        if src_str in [v['name'] for v in cpg.vs] and tgt_str in [v['name'] for v in cpg.vs]:
            if not cpg.are_connected(src_str, tgt_str):
                e = cpg.add_edge(src_str, tgt_str, type=edge_type, curved=False)
    for attr in REQUIRED_VERTEX_ATTRIBUTES:
        if attr not in cpg.vertex_attributes():
            cpg.vs[attr] = [None] * len(cpg.vs)
    for attr in REQUIRED_EDGE_ATTRIBUTES:
        if attr not in cpg.edge_attributes():
            cpg.es[attr] = [False] * len(cpg.es)

    return cpg


def draw_cpg(cpg, output_file="cpg_graph.png", layout="fr"):
    if len(cpg.vs) == 0:
        return False
    for attr in REQUIRED_VERTEX_ATTRIBUTES:
        if attr not in cpg.vertex_attributes():
            cpg.vs[attr] = [None] * len(cpg.vs)
    for attr in REQUIRED_EDGE_ATTRIBUTES:
        if attr not in cpg.edge_attributes():
            cpg.es[attr] = [False] * len(cpg.es)

    try:
        vertex_labels = []
        for v in cpg.vs:
            try:
                label_parts = []
                if "type" in v.attributes() and v["type"]:
                    label_parts.append(v["type"])
                if "location" in v.attributes() and v["location"]:
                    label_parts.append(f"L{v['location']}")
                if "code" in v.attributes() and v["code"]:
                    code = safe_encode(str(v["code"]))
                    if len(code) > 20:
                        code = code[:17] + "..."
                    label_parts.append(code)
                vertex_labels.append("\n".join(label_parts))
            except:
                vertex_labels.append(f"Node {v.index}")
        vertex_colors = []
        for v in cpg.vs:
            try:
                if "type" in v.attributes():
                    if v["type"] == "MethodDeclaration":
                        vertex_colors.append("lightblue")
                    elif v["type"] == "ENTRY":
                        vertex_colors.append("lightgreen")
                    elif "CONTROL" in v["type"]:
                        vertex_colors.append("orange")
                    else:
                        vertex_colors.append("pink")
                else:
                    vertex_colors.append("gray")
            except:
                vertex_colors.append("gray")
        edge_colors = []
        edge_labels = []
        for e in cpg.es:
            try:
                if "type" in e.attributes():
                    e_type = e["type"]
                    if e_type == "DATA":
                        edge_colors.append("red")
                    elif e_type == "CONTROL":
                        edge_colors.append("blue")
                    elif e_type == "CALL":
                        edge_colors.append("green")
                    elif e_type == "FLOW":
                        edge_colors.append("purple")
                    else:
                        edge_colors.append("gray")
                    edge_labels.append(e_type)
                else:
                    edge_colors.append("black")
                    edge_labels.append("EDGE")
            except:
                edge_colors.append("black")
                edge_labels.append("EDGE")

        visual_style = {
            "vertex_size": 30,
            "vertex_label": vertex_labels,
            "vertex_label_size": 8,
            "vertex_color": vertex_colors,
            "edge_color": edge_colors,
            "edge_label": edge_labels,
            "edge_label_size": 7,
            "bbox": (1200, 1200) if len(cpg.vs) < 100 else (2000, 2000),
            "margin": 80,
        }
        if layout == "fr":
            layout = cpg.layout_fruchterman_reingold()
        elif layout == "kk":
            layout = cpg.layout_kamada_kawai()
        else:
            layout = cpg.layout_auto()
        plot(cpg, output_file, layout=layout, **visual_style)
        return True
    except Exception as e:
        traceback.print_exc()
        return False


def save_cpg_to_json(cpg, output_path):
    if cpg is None or len(cpg.vs) == 0:
        return False

    cpg_data = {
        "nodes": [],
        "edges": []
    }

    for v in cpg.vs:
        try:
            node_data = {
                "id": v['name'],
                "type": safe_encode(v['type']) if "type" in v.attributes() else None,
                "location": v['location'] if "location" in v.attributes() else None,
                "code": safe_encode(v['code']) if "code" in v.attributes() else None,
                "filepath": safe_encode(v['filepath']) if "filepath" in v.attributes() else None,
                "method": safe_encode(v['method']) if "method" in v.attributes() else None,
                "class_name": safe_encode(v['class_name']) if "class_name" in v.attributes() else None
            }
            cpg_data["nodes"].append(node_data)
        except KeyError as e:
            print("error")

    for e in cpg.es:
        try:
            source = cpg.vs[e.source]
            target = cpg.vs[e.target]
            edge_data = {
                "source": source['name'],
                "target": target['name'],
                "type": safe_encode(e['type']) if "type" in e.attributes() else None
            }
            cpg_data["edges"].append(edge_data)
        except KeyError as e:
            print("error")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cpg_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        traceback.print_exc()
        return False


def generate_cpg_for_java_file(java_path, output_dir, output_png_dir, output_json_dir, draw_graph=True, save_json=True):
    for root, dirs, files in os.walk(java_path):
        dirs.sort()  
        files.sort() 

        for file in files:
            java_cpg_path = os.path.join(root, file)
            cwe_id = java_cpg_path.split('/')[-2]
            javaCpg_cwe_path = os.path.join(output_dir, cwe_id)
            java_png_cwe_path = os.path.join(output_png_dir, cwe_id)
            java_json_cwe_path = os.path.join(output_json_dir, cwe_id)
            with open(java_cpg_path, 'r', encoding='utf-8') as file:
                java_code = file.read()

            java_file = java_cpg_path.split('/')[-1]
            java_cpg_file = java_file.split('.')[:-1]
            java_cpg_file_1 = '.'.join(java_cpg_file)
            java_cpg_png_path_file = os.path.join(java_png_cwe_path, f"{java_cpg_file_1}_cpg.png") 
            java_cpg_json_path_file = os.path.join(java_json_cwe_path, f"{java_cpg_file_1}_cpg.json") 
            path = os.path.join(javaCpg_cwe_path, java_cpg_file_1)

            if os.path.exists(path):
                continue

            tree = javalang.parse.parse(java_code)
            cpg = build_cpg_from_ast(tree, java_cpg_path)

            with open(path, 'wb') as f:
                pickle.dump(cpg, f)

            if save_json:
                save_cpg_to_json(cpg, java_cpg_json_path_file)

            if draw_graph:
                if not draw_cpg(cpg, java_cpg_png_path_file):
                    try:
                        if len(cpg.vs) > 100:
                            subgraph = cpg.subgraph(cpg.vs[:100])
                            plot(subgraph, java_cpg_png_path_file, bbox=(800, 800))
                    except Exception as e:
                        print(f"error: {e}")


if __name__ == "__main__":
    java_path = '/home/zzb/next_model/datasets/targetVul_java/'
    generate_cpg_for_java_file(
        java_path,
        output_dir='./java_cpg',
        output_png_dir='./java_cpg_png',
        output_json_dir='./java_cpg_json',
        draw_graph=True,
        save_json=True
    )
