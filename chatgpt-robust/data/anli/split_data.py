import json

# 将JSONL文件拆分为300条和900条
def split_jsonl(input_file, output_file_1, output_file_2):
    # 打开输入文件和输出文件
    with open(input_file, 'r') as input_f, open(output_file_1, 'w') as output_f1, open(output_file_2, 'w') as output_f2:
        # 读取输入文件的所有行
        all_lines = input_f.readlines()
        # 获取前300行
        first_300_lines = all_lines[:300]
        # 获取300到900行
        lines_301_to_900 = all_lines[300:]
        # 将前300行写入第一个输出文件
        for line in first_300_lines:
            output_f1.write(line)
        
        for line in lines_301_to_900:
            output_f2.write(line)
        

# 测试代码
input_file = '/home/v-xixuhu/new-chat/chatgpt-robust/data/anli/test.jsonl'
output_file_1 = '/home/v-xixuhu/new-chat/chatgpt-robust/data/anli/test300.jsonl'
output_file_2 = '/home/v-xixuhu/new-chat/chatgpt-robust/data/anli/test900.jsonl'
split_jsonl(input_file, output_file_1, output_file_2)
