import tensorflow as tf
import numpy as np
from model import KGCN
import math
import statistics

def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings2(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_rmse = ctr_eval(sess, model, train_data, args.batch_size)
            eval_rmse = 0#ctr_eval(sess, model, eval_data, args.batch_size)
            test_rmse = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train rmse: %.4f    eval rmse: %.4f    test rmse: %.4f'
                  % (step, train_rmse, eval_rmse, test_rmse))

            # top-K evaluation
            """
            if show_topk:
                rmse = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('rmse: ', end='')
                for i in rmse:
                    print('%.4f\t' % i, end='')
                print('\n')
            """
            
           


def topk_settings2(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        # train/test record: {user: [(item, rating), ...], ....}
        train_record = get_user_record2(train_data)
        test_record = get_user_record2(test_data)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5

def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict

def ctr_eval(sess, model, data, batch_size):
    start = 0
    rmse_list = []
    while start + batch_size <= data.shape[0]:
        rmse = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        rmse_list.append(rmse)
        start += batch_size
    return float(np.mean(rmse_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    rmse_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall

def accuracy_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    rmse_list = []

    for user in user_list:
        test_item_list = [x[0] for x in test_record[user]]
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        """
        for item, label in test_record[user]:
            prediction = item_score_map[item]
            print("label:", label, " prediction:", prediction)
            break
        """
        for item, label in test_record[user]:
            prediction = item_score_map[item]
            rmse_list.append((label - prediction)**2)

    return math.sqrt(statistics.mean(rmse_list))


        
    
def get_user_record2(data):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if user not in user_history_dict:
            user_history_dict[user] = []
        user_history_dict[user].append((item, label))
    return user_history_dict

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict