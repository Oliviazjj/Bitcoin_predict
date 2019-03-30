import csv

def month_string_to_number(string):
    m = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr':4,
         'may':5,
         'jun':6,
         'jul':7,
         'aug':8,
         'sep':9,
         'oct':10,
         'nov':11,
         'dec':12
        }
    s = string.strip()[:3].lower()

    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a month')

with open('gold_price.csv', 'r') as inp, open('gold_price_edit.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[5] != "-":
        	vol = row[5]
        	vol = vol[:-1]
        	# print("vol is: ", vol)
        	float_vol = float(vol)
        	# print("float_vol: ", str(float_vol))
        	row[5] = float_vol * 1000.00
        	row[5] = str(row[5])
        	row[0] = row[0].split(" ")
        	print("row 0 is ", row[0])
        	row[0][0] = month_string_to_number(row[0][0])
        	row[0][1] = row[0][1][:-1]
        	row[0] = str(row[0][2])+'-'+str(row[0][0])+'-'+str(row[0][1])
        	# row[0] = "-".join(row[0])
        	print("row 0 is ", row[0])
        	# print("vol is ", vol, "float_vol: ", float_vol, "after change: ", row[5])
        	writer.writerow(row)