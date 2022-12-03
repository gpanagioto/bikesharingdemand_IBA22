import json
data1 = '/home/georgep/GitHub/bikesharingdemand_IBA22/ExplorentComponentAndConclusion.ipynb'
data2 = '/home/georgep/GitHub/bikesharingdemand_IBA22/IntroAndPrediction.ipynb'
def WordsCounter(file):
    with open(file) as json_file:
        data = json.load(json_file)

    wordCount = 0
    for each in data['cells']:
        cellType = each['cell_type']
        if cellType == "markdown":
            content = each['source']
            for line in content:
                temp = [word for word in line.split() if "#" not in word] # we might need to filter for more markdown keywords here
                wordCount = wordCount + len(temp)
            
    return wordCount

for file in [data1, data2]:
    print(WordsCounter(file))