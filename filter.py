import os

def findSgfFiles(path):
    sgfFiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.sgf'):
                sgfFiles.append(os.path.join(root, file))
    return sgfFiles

def read_file_safely(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin1') as f:
            return f.read()

allValidFile = []
count = 0

allSgfFiles = findSgfFiles('games')
for sgfFile in allSgfFiles:
    try:
        data = read_file_safely(sgfFile)
        if 'HA[' not in data and 'DT[20' in data:
            allValidFile.append(sgfFile)
    except Exception as e:
        print('Error reading:', sgfFile, e)

    count += 1
    if count % 10000 == 0:
        print('Processed ' + str(count) + ' files')

with open('allValid2.txt', 'w', encoding='utf-8') as allValid:
    for sgfFile in allValidFile:
        allValid.write(sgfFile + '\n')

print('Total:', len(allValidFile))
