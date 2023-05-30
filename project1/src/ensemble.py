f_boost = open("boosting_speranza.csv", "r")
f_svr = open("output_svr_speranza.csv", "r")
f_forest = open("output_forest.csv", "r")
f = open("medione_pesato.csv", "w")

header = f_boost.readline()
f_svr.readline()
f_forest.readline()
f.write(header)

for i in range(776):
    line_boost = f_boost.readline()
    line_svr = f_svr.readline()
    line_forest = f_forest.readline()

    id = line_boost.split(',')[0]
    value_boost = float(line_boost.split(',')[1])
    value_svr = float(line_svr.split(',')[1])
    value_forest = float(line_forest.split(',')[1])

    median = ((1/3 * value_boost) + (2/3 * value_svr))

    f.write(id + "," + str(median) + "\n")


f_boost.close()
f_svr.close()
f.close()
