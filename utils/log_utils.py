import atexit

log_init = False
f = open('output/log.txt', 'w')


def all_done():
    print("process exit after execute atexit funcs")
    global f
    f.close()
    f = None


def log(line):
    global log_init
    global f

    if not log_init:
        log_init = True
        atexit.register(all_done)

    f.writelines([line, '\n'])
