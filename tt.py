import json
dt =json.loads(open('xag.json').read())['label_ix']
for ix in dt:
    new_key = dt[ix]
    v = ix.replace(' ','')
    print(new_key,':',v,',')