# coding=utf-8
import re
import data_process
def processAll(allList,preName,cishu,top):
    print('规律总结处理...')
    oneList=[]
    writeList=[]
    for item in allList:
        s=''
        l=[]
        for itemdata in item:
            l.append(itemdata)
            s=s+itemdata+ ','
        print(s[:-1])
        oneList.append(s[:-1])
        writeList.append(l)

    # data_process.write_in_csv(str(xuexilv)+'-'+str(d_a)+'-'+str(zz)+'-ECMA_'+preName+'_Apriori_16_avg.csv', writeList)
    # data_process.write_in_csv('../myMedicalModel/modelvsECMSR_Apriori/AprioriResults/ECMA_' + preName + '_Apriori_16_avg_0.0001_64_0.001-'+str(cishu)+'.csv', writeList)
    data_process.write_in_csv(
        '../myMedicalModel/load_result-627-uit80-H-top8/ECMA_' + preName + '_Apriori_0.001_128_0.0002-top'+top+'-' + str(cishu) + '.csv', writeList)
    oneSet=list(set(oneList))
    sortList=[]
    for item in oneSet:
        num=oneList.count(item)
        sortList.append([num,item])
    # sortList=sorted(sortList,key=lambda x:x[0],reverse=True)
    #
    # for item in sortList:
    #     print(item)
def processAll_tiaocan(allList,preName,xuexilv,d_a,zz):
    print('规律总结处理...')
    oneList=[]
    writeList=[]
    for item in allList:
        s=''
        l=[]
        for itemdata in item:
            l.append(itemdata)
            s=s+itemdata+ ','
        print(s[:-1])
        oneList.append(s[:-1])
        writeList.append(l)

    data_process.write_in_csv('../myMedicalModel/tiaocan_lr0.01/'+str(xuexilv)+'-'+str(d_a)+'-'+str(zz)+'-ECMA_'+preName+'_Apriori_16_avg_unit.csv', writeList)
    # data_process.write_in_csv('../myMedicalModel/modelvsECMSR_Apriori/AprioriResults/ECMA_' + preName + '_Apriori_16_avg_0.001_64_0.001-final-'+str(cishu)+'.csv', writeList)
    oneSet=list(set(oneList))
    sortList=[]
    for item in oneSet:
        num=oneList.count(item)
        sortList.append([num,item])

def processAll_tiaocan_acc(allList,preName,xuexilv,d_a,zz):
    print('规律总结处理...')
    oneList=[]
    writeList=[]
    print(allList)
    with open('../myMedicalModel/tiaocan_lr0.01/'+str(xuexilv)+'-'+str(d_a)+'-'+str(zz)+'-ECMA_'+preName+'_ACC_unit.csv','w',newline='') as fw:
        import csv
        writer = csv.writer(fw)
        writer.writerows(allList)


def processAll_load_acc(allList,preName,cishu,top):
    print('规律总结处理...')
    oneList=[]
    writeList=[]
    print(allList)
    with open('../myMedicalModel/load_acc-627-uit80-H-top8/ECMA_'+preName+'-top'+top+'-'+str(cishu)+'_0.001_ACC.csv','w',newline='') as fw:
        import csv
        writer = csv.writer(fw)
        writer.writerows(allList)