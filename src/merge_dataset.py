from os import listdir

# Merge eval_set into one single txt

dir = 'eval_set/Sports_Eval/'
all_file = listdir(dir)
text = []
for anyFile in all_file:
    with open(dir + anyFile, 'r') as fp:
        k = fp.read()
        k = k.replace('\n','。')
        k = k.replace(' ', '')
        k = k.replace('　　', '')
        k = k.replace(u'\u3000', '')
        k = k[1:]
        text.append(k)

text = '\n'.join(text)
with open(dir + 'all.txt','w') as fp:
    fp.writelines(text)
