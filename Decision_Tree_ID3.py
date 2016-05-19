import math

__author__ = 'PremSaiKumarReddy Gangana (psreddy@unm.edu)'


def entropy(attributes, data, targetAttr, acctype):
    """
    Calculates Entropy for the given dataset based on targetAttr(Class) and
    for a accuarcy calculation type ( either MisClassfication Impurity
    or Entroppy Impurity
    """
    valFreq = {}
    dataEntropy = 0.0

    i = attributes.index(targetAttr)
    # Calculate the frequency of each of the values in the target attr
    for entry in data:
        if (valFreq.__contains__(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0
    # Calculate the entropy of the data for the target attr
    if(acctype == 'entropyimpurity'):
        for freq in valFreq.values():
            dataEntropy += (-freq / len(data)) * math.log(freq / len(data), 2)
    elif(acctype == 'misclassificationimpurity'):
        dataEntropy += 1 - (max(valFreq.values()) / (len(data)))

    return dataEntropy


def gain(attributes, data, targetAttr, attr, acctype):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    valFreq = {}
    subsetEntropy = 0.0

    # find index of the attribute
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if (valFreq.__contains__(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in valFreq.keys():
        valProb = valFreq[val] / sum(valFreq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * \
            entropy(attributes, dataSubset, targetAttr, acctype)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(attributes, data, targetAttr, acctype) - subsetEntropy)


def majority(attributes, data, target):
    """
    Finds most common value for an attribute
    """
    # find target attribute
    valFreq = {}
    # find target in data
    index = attributes.index(target)
    # calculate frequency of values in target attr
    for tuple in data:
        if (valFreq.__contains__(tuple[index])):
            valFreq[tuple[index]] += 1
        else:
            valFreq[tuple[index]] = 1
    max = 0
    major = ""
    for key in valFreq.keys():
        if valFreq[key] > max:
            max = valFreq[key]
            major = key
    return major


def getValues(data, attributes, attr):
    """
    Gets values in the column of the given attribute
    """
    index = attributes.index(attr)
    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values


def getExamples(data, attributes, best, val):
    """
    Gets values in the column of the given attribute
    """
    examples = [[]]
    index = attributes.index(best)
    for entry in data:
        # find entries with the given value
        if (entry[index] == val):
            newEntry = []
            # add value if it is not in best column
            for i in range(0, len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            examples.append(newEntry)
    examples.remove([])
    return examples


def chooseAttr(data, attributes, target, acctype):
    """
    Choose best attibute
    """
    best = attributes[0]
    maxGain = 0
    for attr in attributes:
        if(attr != "Class"):
            newGain = gain(attributes, data, target, attr, acctype)
            if newGain > maxGain:
                maxGain = newGain
                best = attr
    return best


def makeTree(data, attributes, target, recursion, acctype):
    """ Iterates through this function recursively over best attribute
    and sub dataset"""
    recursion += 1
    # Returns a new decision tree based on the examples given.
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majority(attributes, data, target)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = chooseAttr(data, attributes, target, acctype)
        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best: {}}
        chisquaretest(data, attributes, target, best, recursion)
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            examples = getExamples(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = makeTree(examples, newAttr, target, recursion, acctype)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree


def getcountofclassval(examples, attributes, target):
    """ Gets the count of the Promoters and
    non-promoters for a given attribute"""
    valFreq = {}
    i = attributes.index(target)
    # Calculate the frequency of each of the values in the target attribute
    for entry in examples:
        if (valFreq.__contains__(entry[i])):
            valFreq[entry[i]] += 1.0
        else:
            valFreq[entry[i]] = 1.0
    return valFreq


def getrecords(data, attributes, best, val):
    """ Gets the records for the given attribute
    and value of the attribute """
    index = attributes.index(best)
    examples = []
    for record in data:
        if(record[index] == val):
            examples.append(record)
    return examples


def chisquaretest(data, attributes, target, best, recursion):
    """ Takes the dataset, attributes, target, node attribute
    and gives the chisqaure value"""
    rootvalFreq = getcountofclassval(data, attributes, target)
    no_of_rootrecords = sum(rootvalFreq.values())
    chivalue = 0
    values = getValues(data, attributes, best)
    for val in values:
        examples = getrecords(data, attributes, best, val)
        valFreq = getcountofclassval(examples, attributes, target)
        valFreqp = valFreq.get('+', 0)
        valFreqn = valFreq.get('-', 0)
        valFreqtotal = valFreqp + valFreqn
        expectp = (valFreqtotal * (rootvalFreq.get('+', 0))) / \
            no_of_rootrecords
        expectn = (valFreqtotal * (rootvalFreq.get('-', 0))) / \
            no_of_rootrecords
        chivalue += (math.pow((valFreqp - expectp), 2)) / expectp
        chivalue += (math.pow((valFreqn - expectn), 2)) / expectn
    return chivalue


def validatedata(record, att_list, tree, node_list, result):
    """ Validates each record of validating dataset
    and returns the result of the decision tree """
    keys = str(tree.keys())
    keyatt = ""
    keyattval = ""
    keys = keys[12:]
    for letter in keys:
        if((letter == "]" or letter == ")" or letter == "'" or letter == " ")):
            pass
        else:
            keyatt = keyatt + letter
    if(len(keyatt) <= 7):
        keyattval = keyatt.split(',')
        i = att_list.index(node_list[-1])
        if(record[i] in keyattval):
            tree = tree[record[i]]
            if(isinstance(tree, str)):
                result.append(tree)
                del node_list[:]
            else:
                validatedata(record, att_list, tree, node_list, result)
        else:
            result.append('-')
            del node_list[:]
    else:
        node_list.append(keyatt)
        validatedata(record, att_list, tree[keyatt], node_list, result)
    return


def formatdata(filename):
    """ Gives formatted data output for specified file """
    data = open(filename)
    data_list = []
    skipindex = 58
    if(filename == 'training.txt'):
        length_data = (60 * 71) - 1
    else:
        length_data = (60 * 35) - 1

    for index in range(length_data):
        seq = data.read(1)
        if (data.tell() in (skipindex, skipindex + 2)):
            pass
            if(data.tell() == skipindex + 2):
                skipindex += 60
        else:
            data_list.append(seq)

    data.close()

    formatted_data = []
    for index in range(0, len(data_list), 58):
        formatted_data.append(data_list[index:(index + 58)])

    return formatted_data


def accuracy(act_result, exp_result):
    """ Calculates the accuracy of the
    decision tree on validation set """
    tp = tn = fp = fn = 0
    for i in range(len(act_result)):
        if(exp_result[i] == '+'):
            if(act_result[i] == '+'):
                tp += 1
            elif(act_result[i] == '-'):
                fn += 1
        elif(exp_result[i] == '-'):
            if(act_result[i] == '+'):
                fp += 1
            elif(act_result[i] == '-'):
                tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


def main():
    """ Main function which
    controls the flow of program"""
    # training and validation datasets are foramtted
    # such that it can be used further in the program
    train_data = formatdata('training.txt')
    valid_data = formatdata('validation.txt')

    # attributelist is created with values
    # in the form "Attribute<index>" eg: Attribute16
    att_list = []
    for att_index in range(0, 57):
        att_list.append("Attribute" + str(att_index))

    # appends the target class name : 'Class'
    att_list.append('Class')
    target = 'Class'

    # accuracy type to be used in the program
    acctypelist = ["misclassificationimpurity", "entropyimpurity"]
    for acctype in acctypelist:
        # generation of tree
        tree = makeTree(train_data, att_list, target, 0, acctype)

        # validation of tree with validation set
        node_list = []
        result = []
        for record in valid_data:
            validatedata(record, att_list, tree, node_list, result)
            node_list = []
        actual_result = result #collecting the predicted labels from decision tree

        # actual class of the validationset
        expected_result = []
        target_index = att_list.index(target)
        for record in valid_data:
            expected_result.append(record[target_index])
        print(acctype, "accuracy :", accuracy(actual_result, expected_result))


if __name__ == '__main__':
    main()
