def main():
    print("fgfhg")
    f_test = open("test.csv","r")
    f_out = open("out.csv","w")

    header = f_test.readline()
    f_out.write("Id,y\n")
    for line in f_test:
        res = line.split(",")
        id = res[0]
        sum = 0
        for i in range(1,len(res)):
            sum = sum + float(res[i])
        sum = sum / 10

        toWrite = str(id) + "," +str(sum) + "\n"
        f_out.write(toWrite)

if __name__ == "__main__":
    main()
