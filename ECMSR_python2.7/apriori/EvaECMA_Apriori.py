# coding=utf-8
"""
Description     : Simple Python implementation of the Apriori Algorithm
Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""
import chardet
import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from data5W_formula_function import data_process


import codecs

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        # print 'localSet',localSet
        for item, count in localSet.items():
                # print item,count
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        # print 'record',record
        transaction = frozenset(record)
        # print 'transaction', transaction
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets，即生成1-频繁项集
    # print 'zz',itemSet
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])
    toRetRules = []
    return toRetItems, toRetRules


def printResults(items,writecsvname):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    medicallist=[]
    count=0
    for item, support in sorted(items, key=lambda (item, support): support):
        medical=''
        for data in item:
            medicallist.append(data)
            medical=medical+data+'+'
        str1="#%d# item:_%s_%.3f" % (count,medical, support)
        # print str1
        data_process.write_str_in_csv_a(writecsvname, str1)
        # print "#%d# item: %s , %.3f" % (count,item, support)
        count+=1

def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = codecs.open(fname, 'rU','gbk')
        # file_iter = codecs.open(fname, 'rU','utf8')
        for line in file_iter:
                line = line.strip().rstrip(',')                    # Remove trailing comma
                record = frozenset(line.split(','))
                yield record
def printTowrite(words):
    # print words
    data_process.write_str_in_csv_a(writeCsv, words)

if __name__ == "__main__":
    #手动修改名称 start
    import os
    print os.curdir
    preNameList = ['QRJD', 'HXHY', 'HT', 'ZJG', 'ZX', 'XZ', 'LS', 'MM', 'JP', 'ZK', 'AS', 'HXZT', 'ZY', 'TL']
    for preName in preNameList:
        print '************************************************ 功效：%s **********************************************************'%(preName)
        # group10Count = [1,2,3,4,5,6,7,8]
        group10Count = [1]
       # 评估的药组味数，这时是评估三味药和五味药
        evajiweiyaoList = [1,2,3,4,5,6]
       # 手动修改名称 end
         # ECMA_'+preName+'_Apriori_16.csv
        filename='../ECMA_Apriori_10_20epochs_0.01_l2_1/ECMA_%s_Apriori_1.csv'%(preName)
        writecsvname = '../ECMA_aprioriResults/%s_0618.csv' % preName
        writeCsv = '../aprioriEvaDice/0618_%s_diceEvaluate.csv' % preName
        #group47Count存放对应mlt模型中4,7隐藏单元得到的药组个数，这里是第4,7隐藏单元
        group47Count=[]
        group47Count.append(group10Count[0])

        # part 1 start
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                              dest='input',
                              help='filename containing csv',
                              default=filename)
        optparser.add_option('-s', '--minSupport',
                              dest='minS',
                              help='minimum support value',
                              default=0.008,
                              type='float')
        optparser.add_option('-c', '--minConfidence',
                              dest='minC',
                              help='minimum confidence value',
                              default=1,
                              type='float')

        (options, args) = optparser.parse_args()

        inFile = None
        if options.input is None:
                 inFile = sys.stdin
        elif options.input is not None:
                 inFile = dataFromFile(options.input)
        else:
                 print 'No dataset filename specified, system with exit\n'
                 sys.exit('System will exit')

        minSupport = options.minS
        minConfidence = options.minC

        print 'minSupport,minConfidence:',minSupport,minConfidence
        print inFile
        items, rules = runApriori(inFile, minSupport, minConfidence)

        printResults(items,writecsvname)
        # part 1 end

        # print '*************** 功效%s 在mlt model中 【4】个隐藏单元能找到【 %s】组功效配伍 ，【 7】个隐藏单元能找到【 %s】组功效配伍' %(function,group47Count[0],group47Count[1])
       # part 2 start
        #开始评估....
        print '获取Apriori结果中..........'
        aprioriData = data_process.read_csv(writecsvname)
        aprioriList=[]

        for item in aprioriData:
            # print item
            mstr=item[0].split('_')[1]
            perList = []
            # print 'mstr',mstr
            perList=mstr.split('+')
            perList.pop(-1)
            # print 'perList',perList
            aprioriList.append(perList)

        aprioriEvaList=[]
        # print len(aprioriList)
        allLen=len(aprioriList)
        for goupnum in group47Count:
            for i in range(1,6):
               gnum=0
               for j in range(0,allLen)[::-1]:
                   # print aprioriList[j]
                   if len(aprioriList[j])==i:
                       aprioriEvaList.append(aprioriList[j])
                       gnum+=1
                       if gnum==goupnum:
                           break
        print  '获取 i 频繁项集最高支持度的item完毕....'
        words = '*********** 最大频繁项集为 : %d'%len(aprioriEvaList)
        # print words
        data_process.write_str_in_csv_a(writeCsv, words)
        # words ='*********** 依次输出找到的频繁项集'
        # print words
        data_process.write_str_in_csv_a(writeCsv, words)

        for item in aprioriEvaList:
                medicalstr=''
                for itemdata in item:
                    medicalstr=medicalstr+itemdata+','
                words ='支持度 最高的 %d - 频繁项集 为 : %s'%(len(item),medicalstr)
                print words
                data_process.write_str_in_csv_a(writeCsv, words)

        words ='########################## medical evaluating.... ##########################'
        printTowrite(words)
        evalueatecsv = '../%sFile/%s_evaluate.csv' % (preName, preName)
        evalueateData = data_process.read_csv(evalueatecsv)
        evalueateDataList = []
        for item in evalueateData:
            item[0] = item[0].replace('﻿', '')
            for i in range(len(item)):
                item[i] = item[i].decode('utf8')
            evalueateDataList.append(item)
        for item in evalueateDataList:
            zstr=''
            for itemset in item:
                zstr=zstr+itemset+','
            # print zstr
        finalaprioriEvaList=[]
        for i,goupnum in enumerate(group47Count):
            if i==0:
                lastNum=0
            finalaprioriEvaList=aprioriEvaList[lastNum:lastNum+goupnum*5]
            lastNum=lastNum+goupnum*5
            # print 'group'
            words ='goupnum:%s'%goupnum
            printTowrite(words)
            for item in finalaprioriEvaList:
                medicalstr = ''
                for itemdata in item:
                    medicalstr = medicalstr + itemdata + ','
                words = '支持度较高的 %d - 频繁项集 为 : %s' % (len(item), medicalstr)
                printTowrite(words)
            checklist = []
            for itemSet in finalaprioriEvaList:
                finalValue=0
                calN=len(itemSet)
                for stanitem in evalueateDataList:
                    calstanNum=len(stanitem)
                    checkstr = ''
                    calnum = 0
                    for stanData in stanitem:
                        stanData = stanData.decode('gbk',errors='ignore')
                        checkstr = checkstr + stanData + ','
                        for itemData in itemSet:
                            itemData = itemData.decode('gbk',errors='ignore')
                            if stanData.find(itemData)>-1:
                                calnum+=1
                                break
                        value=round(2*float(calnum)/(len(itemSet)+calstanNum),4)
                    if value>finalValue:
                        finalValue=value
                checklist.append(finalValue)
            if goupnum==1 or goupnum==0:
                allAvgDice=sum(i for i in checklist)/5
                print '1组或0组 Apriori算法  1~5 味药 的得分分别为：', checklist, allAvgDice
                words =  '1组或0组 Apriori算法  1~5 味药 的总平均dice得分为 %s 不同长度药组dice的得分分别为：'% allAvgDice
                printTowrite(words)
                data_process.write_str_in_csv_a(writeCsv,checklist)
            else:
                finalChecklist = []
                fnum = 0
                for i,item in enumerate(checklist):
                    fnum=fnum+item
                    if (i+1)%goupnum==0:
                        # print '单组药得分',fnum
                        finalChecklist.append(round(fnum/goupnum,4))
                        fnum=0
                allAvgDice=sum(i for i in finalChecklist)/5
                print '大于1组 Apriori算法  1~5 味药 的平均Dice得分分别为：', finalChecklist,allAvgDice
                words =  '大于1组 Apriori算法  1~5 味药 的总平均dice得分为 %s 不同长度药组dice的得分分别为：'%allAvgDice
                printTowrite(words)
                data_process.write_str_in_csv_a(writeCsv,finalChecklist)
                print 'checkList',checklist
                data_process.write_str_in_csv_a(writeCsv,checklist)






