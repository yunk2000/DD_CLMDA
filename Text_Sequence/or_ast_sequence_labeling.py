import shutil
import os
import xml.etree.ElementTree as ET

def main():
    path_data = '/home/zzb/next_model/datasets/'
    ast_sen_data = './ast_sequence_data/'
    vul_line = '/home/zzb/next_model/datasets/origin_dataset.xml'
    merge_data = 'merge_data/ast_data.txt'

    if os.path.exists("merge_data"):
        shutil.rmtree("merge_data")
    os.mkdir("merge_data")

    tree = ET.parse(vul_line)
    root = tree.getroot()

    vul_info = {}
    for testcase in root.findall('testcase'):
        CWE_ID = testcase.attrib['CWE_ID']
        target = testcase.attrib['target']

        file_element = testcase.find('file')
        file_path = file_element.attrib.get('path')
        file_path = file_path[2:]

        merge_file_path = os.path.join(path_data, file_path)
        m_l = merge_file_path.split('\\')
        merge_file_path = "/".join(m_l)

        flaw_element = file_element.find('flaw')
        flaw_line = flaw_element.attrib['line']
        flaw_line = flaw_line.strip()[1:-1]
        flaw_line = flaw_line.split(', ')
        lines = [int(line) for line in flaw_line]

        if merge_file_path not in vul_info.keys():
            vul_info[merge_file_path] = lines
        else:
            old_lines = vul_info[merge_file_path]
            for i in lines:
                old_lines.append(i)
            vul_info[merge_file_path] = old_lines

    file_count = 0
    for root, dirs, files in os.walk(ast_sen_data):
        dirs.sort()
        files.sort()

        for file in files:
            file_label = 0
            ast_sen_path = os.path.join(root, file)

            with open(ast_sen_path, 'r') as file:
                sen_contents = file.readlines()

            sen_path = sen_contents[0]
            sen_path = sen_path.strip()

            sen_lines = []
            ast_sens = []
            sen_contents = sen_contents[1:]
            for sen_content in sen_contents:
                sen_content = sen_content.strip()

                sen_content_list = sen_content.split(" ")
                sen_line = int(sen_content_list[-1])
                if sen_line not in sen_lines:
                    sen_lines.append(sen_line)

                sen_con_list = sen_content_list[:-1]
                sen_con = " ".join(sen_con_list)
                sen_con_n = sen_con + '\n'
                ast_sens.append(sen_con_n)

            vul_lines = vul_info[sen_path]
            for s_line in sen_lines:
                if s_line in vul_lines:
                    file_label = 1
                    break

            s = ''.join(ast_sens)
            file_count += 1
            with open(merge_data, 'a') as txt_file:
                txt_file.write(str(file_count) + ' ' + sen_path + '\n')
                txt_file.write(s)
                txt_file.write(str(file_label) + '\n')
                txt_file.write('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

if __name__ == '__main__':
    main()

