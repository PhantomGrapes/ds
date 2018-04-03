#coding:utf-8

f = open("../data/train.txt", 'r', encoding='utf-8')
lens = []
s_lens = []
for line in f:
    line = line.strip()
    lens.append(len(line))
    if len(s_lens) == 0:
        s_lens.append(0)
    if line == '':
        s_lens.append(0)
    else:
        s_lens[-1] += 1
print(min(lens), max(lens), sum(lens)/len(lens))
print(min(s_lens), max(s_lens), sum(s_lens)/len(s_lens))