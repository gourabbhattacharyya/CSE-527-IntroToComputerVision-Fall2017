
#domain = [[1 for x in range(7)] for x in range(4)]
domain = []
for x in range(4):
    tempList = []
    for x in range(7):
        tempList.append(1)
    domain.append(tempList) 


print "Domain Value", domain