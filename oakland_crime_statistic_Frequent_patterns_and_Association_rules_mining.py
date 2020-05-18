import pandas as pd
from progressbar import *
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


min_sup = 0.1
min_conf = 0.5


Property_list = ['location', 'Area Id', 'beat', 'Priority', 'Incident Type Id', 'Event Number']

class Association_rules():
    def __init__(self):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def apriori(self, dataset):             #算法主体
        C1 = self.C1_generation(dataset)        #生成单元数候选项集
        dataset = [set(data) for data in dataset]
        F1, sup_rata = self.Ck_low_support_filtering(dataset, C1)
        F = [F1]
        k = 2
        while len(F[k-2]) > 0:
            Ck = self.apriori_gen(F[k-2], k)        #当候选项元素大于2时，合并时检测是否子项集满足频繁
            Fk, support_k = self.Ck_low_support_filtering(dataset, Ck)      #过滤支持度低于阈值的项集
            sup_rata.update(support_k)
            F.append(Fk)
            k += 1
        return F, sup_rata

    def C1_generation(self, dataset):       #生成单元数候选项集
        C1 = []
        progress = ProgressBar()
        for data in progress(dataset):
            for item in data:
                if [item] not in C1:
                    C1.append([item])
        return [frozenset(item) for item in C1]

    def Ck_low_support_filtering(self, dataset, Ck):        #过滤支持度低于阈值的项集
        Ck_count = dict()
        for data in dataset:
            for cand in Ck:
                if cand.issubset(data):
                    if cand not in Ck_count:
                        Ck_count[cand] = 1
                    else:
                        Ck_count[cand] += 1

        num_items = float(len(dataset))
        return_list = []
        sup_rata = dict()
        # 过滤非频繁项集
        for key in Ck_count:
            support  = Ck_count[key] / num_items
            if support >= self.min_sup:
                return_list.insert(0, key)
            sup_rata[key] = support
        return return_list, sup_rata

    def apriori_gen(self, Fk, k):       #当候选项元素大于2时，合并时检测是否子项集满足频繁
        return_list = []
        len_Fk = len(Fk)

        for i in range(len_Fk):
            for j in range(i+1, len_Fk):
                # 第k-2个项相同时，将两个集合合并
                F1 = list(Fk[i])[:k-2]
                F2 = list(Fk[j])[:k-2]
                F1.sort()
                F2.sort()
                if F1 == F2:
                    return_list.append(Fk[i] | Fk[j])
        return return_list

    def generate_rules(self, F, sup_rata):
        """
        产生强关联规则算法实现
        基于Apriori算法，首先从一个频繁项集开始，接着创建一个规则列表，
        其中规则右部只包含一个元素，然后对这些规则进行测试。
        接下来合并所有的剩余规则列表来创建一个新的规则列表，
        其中规则右部包含两个元素。这种方法称作分级法。
        :param F: 频繁项集
        :param sup_rata: 频繁项集对应的支持度
        :return: 强关联规则列表
        """
        strong_rules_list = []
        for i in range(1, len(F)):
            for freq_set in F[i]:
                H1 = [frozenset([item]) for item in freq_set]
                # 只获取有两个或更多元素的集合
                if i > 1:
                    self.rules_from_reasoned_item(freq_set, H1, sup_rata, strong_rules_list)
                else:
                    self.cal_conf(freq_set, H1, sup_rata, strong_rules_list)
        return strong_rules_list

    def rules_from_reasoned_item(self, freq_set, H, sup_rata, strong_rules_list):
        """
        H->出现在规则右部的元素列表
        """
        m = len(H[0])
        if len(freq_set) > (m+1):
            Hmp1 = self.apriori_gen(H, m+1)
            Hmp1 = self.cal_conf(freq_set, Hmp1, sup_rata, strong_rules_list)
            if len(Hmp1) > 1:
                self.rules_from_reasoned_item(freq_set, Hmp1, sup_rata, strong_rules_list)

    def cal_conf(self, freq_set, H, sup_rata, strong_rules_list):          #评估规则
        prunedH = []
        for reasoned_item in H:
            sup = sup_rata[freq_set]
            conf = sup / sup_rata[freq_set - reasoned_item]
            lift = conf / sup_rata[reasoned_item]
            jaccard = sup / (sup_rata[freq_set - reasoned_item] + sup_rata[reasoned_item] - sup)
            if conf >= self.min_conf:
                strong_rules_list.append((freq_set-reasoned_item, reasoned_item, sup, conf, lift, jaccard))
                prunedH.append(reasoned_item)
        return prunedH



class Oakland_Crime_Statistics():
    def __init__(self):
        # 结果文件路径
        self.result_path = './results'

        pass

    def data_read(self):

        data2011 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2011.csv", encoding="utf-8")
        data2012 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2012.csv", encoding="utf-8")
        data2013 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2013.csv", encoding="utf-8")
        data2014 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2014.csv", encoding="utf-8")
        data2015 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2015.csv", encoding="utf-8")
        data2016 = pd.read_csv("./data/Oakland-Crime-Statistics-2011-to-2016/records-for-2016.csv", encoding="utf-8")

        """
        print("2011数据集有以下属性", data2011.columns)
        print("2012数据集有以下属性", data2012.columns)
        print("2013数据集有以下属性", data2013.columns)
        print("2014数据集有以下属性", data2014.columns)
        print("2015数据集有以下属性", data2015.columns)
        print("2016数据集有以下属性", data2016.columns)
        """

        data2012.rename(columns={"Location 1": "Location"}, inplace = True)
        data2013.rename(columns={"Location ": "Location"}, inplace = True)
        data2014.rename(columns={"Location 1": "Location"}, inplace = True)

        data2011_temp = data2011[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
        data2012_temp = data2012[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
        data2013_temp = data2013[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
        data2014_temp = data2014[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
        data2015_temp = data2015[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
        data2016_temp = data2016[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]

        data_all = pd.concat([data2011_temp, data2012_temp, data2013_temp, data2014_temp, data2015_temp, data2016_temp],
                             axis=0)
        print("综合数据集有以下属性", data_all.columns)
        data_all = data_all.dropna(how='any')

        return data_all.head(50000)
        #return data_all


    def mining(self, feature_list):
            out_path = self.result_path
            association = Association_rules()

            data_all = self.data_read()
            rows = data_all.values.tolist()

            # 将数据转为数据字典存储

            dataset = []
            feature_names = ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]
            for data_line in rows:
                data_set = []
                for i, value in enumerate(data_line):
                    if not value:
                        data_set.append((feature_names[i], 'NA'))
                    else:
                        data_set.append((feature_names[i], value))
                dataset.append(data_set)

            # 获取频繁项集
            freq_set, sup_rata = association.apriori(dataset)
            sup_rata_out = sorted(sup_rata.items(), key=lambda d: d[1], reverse=True)
            print("sup_rata ", sup_rata)
            # 获取强关联规则列表
            strong_rules_list = association.generate_rules(freq_set, sup_rata)
            strong_rules_list = sorted(strong_rules_list, key=lambda x: x[3], reverse=True)
            print("strong_rules_list ", strong_rules_list)

            # 将频繁项集输出到结果文件
            freq_set_file = open(os.path.join(out_path, '频繁项集.json'), 'w')
            for (key, value) in sup_rata_out:
                result_dict = {'set': None, 'sup': None}
                set_result = list(key)
                sup_result = value
                if sup_result < min_sup:
                    continue
                result_dict['set'] = set_result
                result_dict['sup'] = sup_result
                json_str = json.dumps(result_dict, ensure_ascii=False)
                freq_set_file.write(json_str + '\n')
            freq_set_file.close()

            # 将关联规则输出到结果文件
            rules_file = open(os.path.join(out_path, '规则.json'), 'w')
            for result in strong_rules_list:
                result_dict = {'X_set': None, 'Y_set': None, 'sup': None, 'conf': None, 'lift': None, 'jaccard': None}
                X_set, Y_set, sup, conf, lift, jaccard = result
                result_dict['X_set'] = list(X_set)
                result_dict['Y_set'] = list(Y_set)
                result_dict['sup'] = sup
                result_dict['conf'] = conf
                result_dict['lift'] = lift
                result_dict['jaccard'] = jaccard

                json_str = json.dumps(result_dict, ensure_ascii=False)
                rules_file.write(json_str + '\n')
            rules_file.close()



if __name__ == "__main__":
    Oakland_Crime_Statistics().mining(Property_list)
