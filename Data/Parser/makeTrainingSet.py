import json
import os
import re

import nltk


def checkJaccadSimilarity(querytitle, corpusTitles, threshold=0.95):
    for title in corpusTitles:
        jd = nltk.jaccard_distance(set(querytitle), set(title))
        jsim = 1 - jd
        if jsim > threshold:
            # found another title which is similar
            if title != querytitle:
                print('Original:', title)
                print('Query:', querytitle)
                print()
            return True
    return False


print('Preprocessing')
wetlabsdir = '/Users/Admin/Documents/UMassMSCS/Courses/692/692Project/Data/WLP-Dataset/protocol-data'
protocolsdir = '/Users/Admin/Documents/UMassMSCS/Courses/692/692Project/Data/protocols/data'

wetlabsfiles = list(set([x.split('.')[0] for x in os.listdir(wetlabsdir) if
                  re.match('protocol_[\d]*.(ann|txt)', x) ]))

protocolsfiles = os.listdir(protocolsdir)

print('Catching titles for', len(wetlabsfiles), 'wetlabs files')
titles = []
for i, fil in enumerate(wetlabsfiles):
    with open(os.path.join(wetlabsdir, '%s.txt'%fil), 'r') as f:
        protocol = f.read()
        titles.append(protocol.split("\n")[0].lower())


print('Found', len(protocolsfiles),'protocols files')
print('Size of titles list', len(titles))

stringMatch_count = 0
jaccardSim_count = 0

for i, fil in enumerate(protocolsfiles):
    with open(os.path.join(protocolsdir, fil), "r") as read_file:
        data = json.load(read_file)
        queryTitle = data['title'].lower()
        if queryTitle in titles:
            stringMatch_count += 1
        if checkJaccadSimilarity(queryTitle, titles):
            jaccardSim_count += 1

print(stringMatch_count, jaccardSim_count)