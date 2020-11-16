# coding=utf-8
"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
from data5W_formula_function import data_process
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from data5W_formula_function import data_process


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
    # for key, value in largeSet.items()[1:]:
    #     for item in value:
    #         _subsets = map(frozenset, [x for x in subsets(item)])
    #         for element in _subsets:
    #             remain = item.difference(element)
    #             if len(remain) > 0:
    #                 confidence = getSupport(item)/getSupport(element)
    #                 if confidence >= minConfidence:
    #                     toRetRules.append(((tuple(element), tuple(remain)),
    #                                        confidence))
    return toRetItems, toRetRules


def printResults(items, rules,writecsvname):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    medicallist=[]
    count=0
    for item, support in sorted(items, key=lambda (item, support): support):
        medical=''
        for data in item:
            medicallist.append(data)
            medical=medical+data+','
        str1="#%d# item: %s %.3f" % (count,medical, support)
        print str1
        data_process.write_str_in_csv_a(writecsvname, str1)
        # print "#%d# item: %s , %.3f" % (count,item, support)
        count+=1
    # print "\n------------------------ RULES:"
    # count = 0
    # for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
    #     pre, post = rule
    #     premedical = ''
    #     for data in pre:
    #         premedical = premedical + data + ','
    #     postmedical=''
    #     for data in post:
    #         postmedical = postmedical + data + ','
    #     str2="##%d## Rule: %s ==> %s , %.3f" % (count,premedical, postmedical, confidence)
    #     print str2
    #     data_process.write_str_in_csv_a(writecsvname, str2)
    #     count += 1
    return medicallist


def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default='Apriori_HXHY_data.csv')
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.05,
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
    items, rules = runApriori(inFile, minSupport, minConfidence)

    writecsvname = '../aprioriResults/test.csv'
    medicallist=printResults(items, rules,writecsvname)
    medicalSet = list(set(medicallist))
    # print 'medicalSet',medicalSet
    print 'Apriori算法找到药物： %d '%len(medicalSet)