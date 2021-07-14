import tensorflow as tf
import numpy as np
from model import KGCN
import math
import statistics
import pandas as pd

def train(args, data, show_loss):
    n_user, n_entity, n_relation = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    adj_entity, adj_relation = data[6], data[7]
    items, users = data[8], data[9]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

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
            eval_rmse = ctr_eval(sess, model, eval_data, args.batch_size)
            test_rmse = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train rmse: %.4f    eval rmse: %.4f    test rmse: %.4f'
                  % (step, train_rmse, eval_rmse, test_rmse))
        
        #save_results(sess, model, get_user_record(test_data), args.batch_size)
        save_predictions(sess, model, list(items), list(users), args.batch_size)
            

def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 500
        k_list = [1, 2, 5, 10, 20, 50, 100]
        # train/test record: {user: {item: rating, ...}, ....}
        train_record = get_user_record(train_data)
        test_record = get_user_record(test_data)
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

def topk_eval(sess, model, user_list, test_record, k_list, batch_size):
    rmse_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(test_record[user].keys())
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

        #item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        #item_sorted = [i[0] for i in item_score_pair_sorted]

        #print(item_score_pair_sorted)
        #print(test_record[user])

        for k in k_list:
            rmse_temp_list = []
            for item in test_record[user]:
                rmse_temp_list.append((item_score_map[item] - test_record[user][item]) ** 2)
            rmse_temp = math.sqrt(np.mean(sorted(rmse_temp_list)[:k]))
            rmse_list[k].append(rmse_temp)

    rmse = [np.mean(rmse_list[k]) for k in k_list]
    return rmse

def save_results(sess, model, test_record, batch_size):
    results = {'user': [],
                'item': [],
                'rating': [],
                'prediction': [],
                'error': []}

    for user in np.random.choice(list(test_record.keys()), size=500, replace=False):
        test_item_list = list(test_record[user].keys())
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

        for item in test_record[user]:
            results['user'].append(user)
            results['item'].append(item)
            results['rating'].append(test_record[user][item])
            results['prediction'].append(item_score_map[item])
            results['error'].append(abs(test_record[user][item] - item_score_map[item]))

    df = pd.DataFrame(results, columns=['user', 'item', 'rating', 'prediction', 'error'])
    df.to_pickle('../data/restaurants/results.pickle')
       
def save_predictions(sess, model, item_list, user_list, batch_size):
    results = {'user': [],
                'item': [],
                'rating': []}

    for user in np.random.choice(user_list, size=100, replace=False):
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: item_list[start:] + [item_list[-1]] * (
                               batch_size - len(item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        for item in item_list:
            results['user'].append(user)
            results['item'].append(item)
            results['rating'].append(item_score_map[item])

    df = pd.DataFrame(results, columns=['user', 'item', 'rating'])
    df.to_pickle('../data/restaurants/predictions.pickle')

def get_user_record(data):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if user not in user_history_dict:
            user_history_dict[user] = dict()
        user_history_dict[user][item] = label
    return user_history_dict