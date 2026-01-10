import shutil
from igraph import *
import os
import pickle
import xml.etree.ElementTree as ET

def main():
    source_data = '/home/zzb/next_model/datasets/targetVul_java/'
    ast_data = './java_ast/'
    output_data = './java_ast_sequence_data/'

    for root, dirs, files in os.walk(ast_data):
        for file in files:
            ast_file_path = os.path.join(root, file)

            with open(ast_file_path, 'rb') as fin:
                ast = pickle.load(fin)

            ast_path = ''
            for vertex in ast.vs:
                ast_path = vertex['filepath']
                break

            cweID = ast_path.split('/')[-2]
            file_name = ast_path.split('/')[-1]
            c_file_path = os.path.join(source_data, cweID, file_name)

            output_file_name = file + '.txt'
            output_path_1 = os.path.join(output_data, cweID)
            output_path = os.path.join(output_path_1, output_file_name)
            if not os.path.exists(output_path_1):
                os.mkdir(output_path_1)

            with open(c_file_path, 'r') as c_file:
                c_content = c_file.readlines()

            with open(output_path, 'a') as txt_file:
                txt_file.write(c_file_path + '\n')

            line_list = []
            for v in ast.vs:
                location = str(v['location'])
                row = location.split(':')[0]

                if v['type'] == 'FunctionDef':
                    with open(output_path, 'a') as txt_file:
                        txt_file.write(v['code'] + ' ' + str(0) + '\n')
                    continue

                if row == 'None':
                    continue
                row = int(row)
                if row not in line_list:
                    line_list.append(row)

            line_list = sorted(line_list)

            txt_list = []
            for i in line_list:
                code = c_content[i-1].strip()
                code_and_line = code + ' ' + str(i) + '\n'
                txt_list.append(code_and_line)

            txt_content = "".join(txt_list)

            with open(output_path, 'a') as txt_file:
                txt_file.write(txt_content)

if __name__ == '__main__':
    main()
