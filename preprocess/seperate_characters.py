import os, re
import pandas as pd
character = 'Sheldon'

raw_corpus_dir = '../raw_corpus/'
result = []
for filename in os.listdir(raw_corpus_dir):
    file = os.path.join(raw_corpus_dir, filename)
    f = open(file, 'r', encoding='utf-8')
    for line in f:
        char = re.findall(r'^\w+', line)
        if len(char) == 1:
            if char[0] == character:
                newline = re.sub(r'^.*:', '', line)
                result.append(newline)

output = os.path.join('../data/', character + '.txt')
f = open(output, 'w')
for line in result:
    f.write("%s" % line)

