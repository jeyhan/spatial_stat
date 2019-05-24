
def log(line):
    f = open('output/log.txt', 'a')
    f.writelines([line])
    f.close()
