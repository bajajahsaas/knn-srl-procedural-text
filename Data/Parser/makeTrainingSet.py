import json
import os
import re


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
        titles.append(protocol.split("\n")[0])


print('Found', len(protocolsfiles),'protocols files')

# for i, fil in enumerate(protocolsfiles):
#     with open(os.path.join(protocolsdir, fil), "r") as read_file:
#         data = json.load(read_file)