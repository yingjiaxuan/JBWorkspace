import sys
if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    txt = {}
    for i in range(n):#录入文章
        word = str(sys.stdin.readline().strip())
        if word in txt:
            txt[word] = int(txt[word])+1
        else:
            txt[word] = 1
    num = 0
    #print (txt)
    for (k,v) in txt.items():
        #print (v/n)
        if v/n >= 0.01:
            num = num +1
    print (num)