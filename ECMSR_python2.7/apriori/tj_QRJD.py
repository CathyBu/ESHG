# -*- coding: utf-8 -*-
"""
Description     : Simple Python implementation of the Apriori Algorithm
Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""
# import chardet
import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
# import data5W_formula_function
# from data5W_formula_function import data_process
import data_process
import itertools
import csv
import time

import codecs
def sup_top_count(sup2,sup3,sup4,sup5,num2,num3,num4,num5):
    result = []
    result.append([1,2,3,4,5])
    res_num = []
    res_num.append('n')
    for sup_res in [sup2,sup3,sup4,sup5]:
        dd = []
        if sup_res==sup2:
            num_res = num2
        elif sup_res==sup3:
            num_res = num3
        elif sup_res==sup4:
            num_res = num4
        elif sup_res==sup5:
            num_res = num5

        save_sup = []      # [0.03,0.02]   #[大,小]
        for j in sup_res:
            if len(save_sup) != 1:
                if j not in save_sup:
                    save_sup.append(j)
            else:
                break
        r1 = []
        for index,k in enumerate(sup_res):
            if k not in save_sup:
                break
            else:
                r1.append(num_res[index])
        if len(r1)!=0:
            last1 = sum(r1)/len(r1)
        else:
            last1 = 0
        dd.append(last1)
        # dd.append(sum(r1) / len(r1) )

        save_sup = []  # [0.03,0.02]   #[大,小]
        for j in sup_res:
            if len(save_sup) != 2:
                if j not in save_sup:
                    save_sup.append(j)
            else:
                break
        r1 = []
        r2 = []
        for index,k in enumerate(sup_res):
            if k not in save_sup:
                break
            elif save_sup.index(k)==0:
                r1.append(num_res[index])
            elif save_sup.index(k)==1:
                r2.append(num_res[index])

        if len(r1)!=0:
            last1 = sum(r1)/len(r1)
        else:
            last1 = 0
        if len(r2)!=0:
            last2 = sum(r2)/len(r2)
        else:
            last2 = 0

        # dd.append((last1+last2)/2)
        if len(r1)+len(r2)==0:
            dd.append(0)
        else:
            dd.append((sum(r1)+sum(r2)) / (len(r1)+len(r2)))

        save_sup = []  # [0.03,0.02]   #[大,小]
        for j in sup_res:
            if len(save_sup) != 3:
                if j not in save_sup:
                    save_sup.append(j)
            else:
                break
        r1 = []
        r2 = []
        r3 = []
        for index,k in enumerate(sup_res):
            if k not in save_sup:
                break
            elif save_sup.index(k)==0:
                r1.append(num_res[index])
            elif save_sup.index(k)==1:
                r2.append(num_res[index])
            elif save_sup.index(k)==2:
                r3.append(num_res[index])
        if len(r1)!=0:
            last1 = sum(r1)/len(r1)
        else:
            last1 = 0
        if len(r2)!=0:
            last2 = sum(r2)/len(r2)
        else:
            last2 = 0
        if len(r3)!=0:
            last3 = sum(r3)/len(r3)
        else:
            last3 = 0
        # dd.append((last1+last2+last3)/3)
        if len(r1)+len(r2)+len(r3)==0:
            dd.append(0)
        else:
            dd.append((sum(r1) + sum(r2)+sum(r3)) /(len(r1) + len(r2)+len(r3)))

        save_sup = []  # [0.03,0.02]   #[大,小]
        for j in sup_res:
            if len(save_sup) != 4:
                if j not in save_sup:
                    save_sup.append(j)
            else:
                break
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        for index, k in enumerate(sup_res):
            if k not in save_sup:
                break
            elif save_sup.index(k) == 0:
                r1.append(num_res[index])
            elif save_sup.index(k) == 1:
                r2.append(num_res[index])
            elif save_sup.index(k) == 2:
                r3.append(num_res[index])
            elif save_sup.index(k) == 3:
                r4.append(num_res[index])

        if len(r1) != 0:
            last1 = sum(r1) / len(r1)
        else:
            last1 = 0
        if len(r2) != 0:
            last2 = sum(r2) / len(r2)
        else:
            last2 = 0
        if len(r3) != 0:
            last3 = sum(r3) / len(r3)
        else:
            last3 = 0
        if len(r4) != 0:
            last4 = sum(r4) / len(r4)
        else:
            last4 = 0
        # print len(r1), len(r2), len(r3), len(r4)
        # dd.append((last1 + last2 + last3 + last4 ) / 4)
        if len(r1)+len(r2)+len(r3)+len(r4)==0:
            dd.append(0)
        else:
            dd.append((sum(r1) + sum(r2)+sum(r3)+sum(r4)) / (len(r1) + len(r2)+len(r3)+len(r4)))

        save_sup = []  # [0.03,0.02]   #[大,小]
        for j in sup_res:
            if len(save_sup) != 5:
                if j not in save_sup:
                    save_sup.append(j)
            else:
                break
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        r5 = []
        for index,k in enumerate(sup_res):
            if k not in save_sup:
                break
            elif save_sup.index(k)==0:
                r1.append(num_res[index])
            elif save_sup.index(k)==1:
                r2.append(num_res[index])
            elif save_sup.index(k)==2:
                r3.append(num_res[index])
            elif save_sup.index(k)==3:
                r4.append(num_res[index])
            elif save_sup.index(k)==4:
                r5.append(num_res[index])

        if len(r1)!=0:
            last1 = sum(r1)/len(r1)
        else:
            last1 = 0
        if len(r2)!=0:
            last2 = sum(r2)/len(r2)
        else:
            last2 = 0
        if len(r3)!=0:
            last3 = sum(r3)/len(r3)
        else:
            last3 = 0
        if len(r4)!=0:
            last4 = sum(r4)/len(r4)
        else:
            last4 = 0
        if len(r5)!=0:
            last5 = sum(r5)/len(r5)
        else:
            last5 = 0

        print len(r1),len(r2),len(r3),len(r4),len(r5)
        res_num.append(str(len(r1))+'-'+str(len(r2))+'-'+str(len(r3))+'-'+str(len(r4))+'-'+str(len(r5)))
        # dd.append((last1 + last2 + last3+last4+last5) / 5)
        if len(r1)+len(r2)+len(r3)+len(r4)+len(r5)==0:
            dd.append(0)
        else:
            dd.append((sum(r1) + sum(r2) + sum(r3) + sum(r4)+sum(r5)) / (len(r1) + len(r2) + len(r3) + len(r4)+len(r5)))

        result.append(dd)

    import numpy as np
    # print result
    result = np.array(result).transpose()
    print result
    print res_num

    return result, [res_num]

def count_result(suplist,checklist,num_n,preName,minSupport):

    num1 = []
    num2 = []
    num3 = []
    num4 = []
    num5 = []
    sup1 = []
    sup2 = []
    sup3 = []
    sup4 = []
    sup5 = []

    for index,v in enumerate(num_n):
        if v==1:
            num1.append(checklist[index])
            sup1.append(suplist[index])
        elif v==2:
            num2.append(checklist[index])
            sup2.append(suplist[index])
        elif v==3:
            num3.append(checklist[index])
            sup3.append(suplist[index])
        elif v==4:
            num4.append(checklist[index])
            sup4.append(suplist[index])
        elif v==5:
            num5.append(checklist[index])
            sup5.append(suplist[index])

    res = []
    if len(num2)!=0:
        r2=sum(num2)/len(num2)
    else:
        r2 = 0
    if len(num3)!=0:
        r3=sum(num3)/len(num3)
    else:
        r3 = 0
    if len(num4)!=0:
        r4 = sum(num4)/len(num4)
    else:
        r4 = 0
    if len(num5)!=0:
        r5 = sum(num5)/len(num5)
    else:
        r5 = 0
    res.append([preName+'-'+str(minSupport)+'-'+str(len(num2))+'-'+str(len(num3))+'-'+str(len(num4))+'-'+str(len(num5)),r2,r3,r4,r5])

    res2 , res3= sup_top_count(sup2,sup3,sup4,sup5,num2,num3,num4,num5)
    # print num1,num2,num3,num4,num5
    # print checklist

    return res , res2, res3

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList_tj(data_iterator):
    transactionList = list()
    itemSet = set()
    m = set()
    tj_dict = {}
    count = 0
    for record in data_iterator:
        # print 'record',record
        # print len(record)
        if len(record)>30:
            continue
        res_r = []
        for i in record:
            if len(str(i))>3:   # 过滤 单个字 药物
                # print i
                # print len(str(i))
                res_r.append(i)
        record = res_r
        count +=1
        if count%10==0:
            print count
        transaction = frozenset(record)
        # print 'transaction', transaction
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets，即生成1-频繁项集

        for i in range(1,6):
            res = itertools.combinations(record,i)
            for ri in list(res):
                if ' '.join(sorted(ri)) not in tj_dict.keys():
                    tj_dict[' '.join(sorted(ri))] = 1
                else:
                    tj_dict[' '.join(sorted(ri))] += 1
                # print ' '.join(sorted(record))
                # print ' '.join(sorted(ri))
                # print ' '.join(ri)
    # for k,v in tj_dict.items():
    #     print k,v

    return tj_dict,count

def tj(data_iter):

    itemSet_dict, tr_n = getItemSetTransactionList_tj(data_iter)

    def getSupport(item,tr_n):
            """local function which Returns the support of an item"""
            return float(item)/tr_n

    toRetItems = []
    for key, value in itemSet_dict.items():
        # print key
        k = key.strip().split()
        # print k
        # print tuple(k)
        toRetItems.append((tuple(k), getSupport(value,tr_n)))
    # toRetRules = []
    # print toRetItems
    # print len(toRetItems)
    # for i in toRetItems:
    #     print i

    return toRetItems

def printResults(items,writecsvname):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    medicallist=[]
    count=0
    result = []

    for item, support in sorted(items, key=lambda (item, support): support):
        medical=''

        for data in item:
            # print data
            # data = data.decode('gbk',errors='ignore')
            # print data
            medicallist.append(data)
            medical=medical+data+'+'
        str1="#%d# item:_%s_%.4f" % (count,medical, support)
        # print str1
        result.append(str(str1))
        # data_process.write_str_in_csv_a(writecsvname, str1)
        # print "#%d# item: %s , %.3f" % (count,item, support)
        count+=1
    with open(writecsvname,'wb') as fw:
        import csv
        writer = csv.writer(fw)
        writer.writerows(result)

def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = codecs.open(fname, 'rU','gbk',errors='ignore')   # 张apriori
        # file_iter =  open(fname, 'rU')   # apriori
        for line in file_iter:
                line = line.strip().rstrip(',')                    # Remove trailing comma
                record = frozenset(line.split(','))
                yield record


def printTowrite(words):
    # print words
    data_process.write_str_in_csv_a(writeCsv, words)

if __name__ == "__main__":
    #手动修改名称 start
    all_r = []
    all_time = []
    # preNameList = [ 'QRJD','HXHY', 'HT', 'ZJG', 'ZX', 'XZ', 'LS', 'MM', 'JP', 'ZK', 'AS', 'HXZT', 'ZY', 'TL']
    preNameList = [ 'QRJD']
    for preName in preNameList:
      with open('log-time.csv','w') as fwt:
          writer = csv.writer(fwt)
          res_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
          all_time.append(preName+res_time)
          print  '当前时间 ',res_time
          writer.writerows(all_time)

      for sup in  [0.0]:   # apriori不同支持度
      # for sup in [ 0.003,0.005,0.008]:  #张
        print '************************************************ 功效：%s **********************************************************'%(preName)
        # group10Count = [1,2,3,4,5,6,7,8]
        group10Count = [1]
       # 评估的药组味数，这时是评估三味药和五味药
        evajiweiyaoList = [1,2,3,4,5,6]
       # 手动修改名称 end
        # ECMA_'+preName+'_Apriori_16.csv

        # filename = '../ECMA_QRJD/ECMA_%s_Apriori_16_avg.csv' % (preName)  # yaopan修改：ECMA结果apriori-60-avg
        import os
        all_path = os.listdir('../ECMA_QRJD_Hnew+H')
        for filename in all_path:
            print(filename)
            file = '../ECMA_QRJD_Hnew+H/'+filename
            writecsvname = 'tj_QRJD_Hnew+H_623/'+filename  # yaopan修改 60

            writeCsv = '../aprioriEvaDice/0618_%s_diceEvaluate.csv' % preName
            #group47Count存放对应mlt模型中4,7隐藏单元得到的药组个数，这里是第4,7隐藏单元
            group47Count=[]
            group47Count.append(group10Count[0])

            # part 1 start
            optparser = OptionParser()
            optparser.add_option('-f', '--inputFile',
                                  dest='input',
                                  help='filename containing csv',
                                  default=file)
            optparser.add_option('-s', '--minSupport',
                                  dest='minS',
                                  help='minimum support value',
                                  # default=0.008,      # 张
                                 # default=0.03,      # apriori
                                 default=sup,
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

            items  = tj(inFile)
            wn = 'bc_QRJD_Hnew+H_623/'+ filename
            with open(wn,'w') as fwn:
                writer = csv.writer(fwn)
                for ts in items:
                    # print ts
                    mmmm = ' '.join(ts[0])
                    ssss = ts[1]
                    writer.writerow([mmmm,ssss])
            printResults(items, writecsvname)
            # part 1 end

            # print '*************** 功效%s 在mlt model中 【4】个隐藏单元能找到【 %s】组功效配伍 ，【 7】个隐藏单元能找到【 %s】组功效配伍' %(function,group47Count[0],group47Count[1])
           # part 2 start
            #开始评估....
            print '获取Apriori结果中..........'
            print writecsvname
            # aprioriData = data_process.read_csv(writecsvname)
            aprioriData =[]
            with open(writecsvname,'rb') as fr:
                import csv
                reader = csv.reader(fr)
                for i in reader:
                    # print ''.join(i)
                    aprioriData.append(''.join(i))
            aprioriList=[]
            suplist = []
            for item in aprioriData:
                suplist.append(item.split('_')[2])
                mstr=item.split('_')[1]
                perList = []
                # print 'mstr',mstr
                perList=mstr.split('+')
                # print perList
                perList.pop(-1)
                # print 'perList',perList
                # for ff in perList:
                #     print ff
                aprioriList.append(perList)

            aprioriEvaList=[]
            # print '总共：',len(aprioriList)

            # for dd in aprioriList:
            #     print dd[0]

            allLen=len(aprioriList)
            # print suplist
            supres =[ i for i in reversed(suplist)]
            suplist = supres
            # print suplist
            for goupnum in group47Count:
                # print goupnum
                # for i in range(1,6):
                   gnum=0
                   results = []
                   for j in range(0,allLen)[::-1]:
                       # print aprioriList[j]
                       # if len(aprioriList[j])==i:
                           aprioriEvaList.append(aprioriList[j])
                           gnum+=1
                           # if gnum==goupnum:
                               # break
            # print  '获取 i 频繁项集最高支持度的item完毕....'
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
                    # print words
                    data_process.write_str_in_csv_a(writeCsv, words)

            words ='########################## medical evaluating.... ##########################'
            printTowrite(words)
            evalueatecsv = '../%sFile/%s_evaluate.csv' % (preName, preName)
            evalueateData = data_process.read_csv(evalueatecsv)
            evalueateDataList = []
            for item in evalueateData:
                item[0] = item[0].replace('﻿', '')
                for i in range(len(item)):
                    # item[i] = item[i].decode('utf8',errors='ignore')
                    item[i] = item[i].decode('utf8')

                evalueateDataList.append(item)
            # print evalueateDataList
            for item in evalueateDataList:
                zstr=''
                for itemset in item:
                    zstr=zstr+itemset+','
                # print zstr
            finalaprioriEvaList=[]
            for i,goupnum in enumerate(group47Count):
                if i==0:
                    lastNum=0
                # finalaprioriEvaList=aprioriEvaList[lastNum:lastNum+goupnum*5]
                finalaprioriEvaList =aprioriEvaList
                lastNum=lastNum+goupnum*5
                # print 'group'
                words ='goupnum:%s'%goupnum
                printTowrite(words)
                for item in finalaprioriEvaList:
                    medicalstr = ''
                    for itemdata in item:
                        medicalstr = medicalstr + itemdata + ','
                    words = '支持度较高的 %d - 频繁项集 为 : %s' % (len(item), medicalstr)
                    # print words
                    printTowrite(words)
                checklist = []
                num_n = []
                for itemSet in finalaprioriEvaList:
                    finalValue=0
                    calN=len(itemSet)
                    for stanitem in evalueateDataList:
                        calstanNum=len(stanitem)
                        checkstr = ''
                        calnum = 0
                        for stanData in stanitem:
                            stanData = stanData.decode('gbk',errors='ignore')
                            # stanData = stanData.decode('utf8')
                            checkstr = checkstr + stanData + ','
                            for itemData in itemSet:
                                # try:
                                    itemData = itemData.decode('gbk',errors='ignore')
                                    # itemData = itemData.decode('utf8')
                                    if stanData.find(itemData) >-1:
                                        calnum+=1
                                        break
                                # except:
                                    # print itemData
                                    # pass
                            value=round(2*float(calnum)/(len(itemSet)+calstanNum),4)
                            # if value==0.8:
                            #     print stanData,itemData,len(stanData),len(itemData)
                            #     print stanitem
                            #     print len(stanitem)
                            #     print itemSet
                            #     print len(itemSet)

                        if value>finalValue:
                            finalValue=value
                    checklist.append(finalValue)
                    num_n.append(len(itemSet))

                if goupnum==1 or goupnum==0:
                    allAvgDice=sum(i for i in checklist)/5
                    # print '1组或0组 Apriori算法  1~5 味药 的得分分别为：', checklist, allAvgDice
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
                    # print 'checkList',checklist
                    data_process.write_str_in_csv_a(writeCsv,checklist)

                # print checklist
                # rr,rr2,rr3 = count_result(suplist,checklist,num_n,preName,sup)
                rr, rr2, rr3 = count_result(suplist, checklist, num_n, str(filename), sup)
                all_r.append(rr)
                all_r.append(rr2)
                all_r.append(rr3)
            all_r.append(['','','',''])
    import csv
    with open('tj_QRJD_Hnew+H_623.csv','wb') as fw:
        writer = csv.writer(fw)
        writer.writerow(['功效-支持度-num2-num3-num4-num5',2,3,4,5])
        for i in all_r:
            writer.writerows(i)

