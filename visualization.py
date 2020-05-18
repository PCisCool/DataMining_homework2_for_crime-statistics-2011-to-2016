import matplotlib.pyplot as plt
import json


class Visualization():

    with open("./results/频繁项集.json") as f1:
        freq = [json.loads(each) for each in f1.readlines()]

    with open("./results/规则.json") as f2:
        rules = [json.loads(each) for each in f2.readlines()]

    freq_sup = [each["sup"] for each in freq]
    plt.boxplot(freq_sup)
    plt.ylabel("Frequent item")
    plt.show()

    rules_sup = [each["sup"] for each in rules]
    rules_conf = [each["conf"] for each in rules]


    plt.scatter(rules_sup, rules_conf, marker='o', color='red', s=40)
    plt.xlabel = 'Sup'
    plt.ylabel = 'Conf'
    plt.legend(loc='best')
    plt.show()

Visualization()
