import re, os
from collections import Counter

major_character = set(['Sheldon', 'Leonard', 'Penny', 'Howard', 'Raj', 'Amy', 'Bernadette'])

dir = '../raw_corpus'
match = []
for filename in os.listdir(dir):
    file = os.path.join(dir, filename)
    f = open(file, 'r')
    for line in f:
        char = re.findall(r'^\w+', line)
        if len(char) == 1:
            if char[0] in major_character:
                match += char
            else:
                match += ['rest']

count = Counter(match)
print(count)
