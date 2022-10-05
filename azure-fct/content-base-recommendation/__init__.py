import logging

import azure.functions as func
import pandas as pd
import numpy as np


def main(req: func.HttpRequest) -> func.HttpResponse:

    # Parse request params
    req_body = req.get_json()
    articles = pd.DataFrame(eval(req_body['articles']))
    distances = pd.DataFrame(eval(req_body['distances']))
    distances.set_index('index', inplace=True)
    articles_ids = np.array(req_body['articles_ids'])
    recommendation_nb = int(req_body['recommendation_nb'])
    user_id = req_body['user_id']

    # Get the indexes of articles
    articles['index'] = [np.where(articles_ids == article_id)[0][0] for article_id in articles['article_id']]
    avoid_indexes = articles['index'].tolist()

    # For each articles read, we get the five closest **that the user did not read yet**
    closests = []
    for i, row in articles.iterrows():
        dists = distances.loc[row['index']].drop(index=avoid_indexes, errors='ignore')
        closest = dists.sort_values(ascending=False)[:recommendation_nb] / row['weight']

        for ind, val in closest.iteritems():
            closests.append({'article_id': articles_ids[int(ind)], 'dist': val, 'coming_from': row['article_id']})

    # Only want the 5 closests
    closests.sort(key=lambda x: x['dist'])
    selection = pd.DataFrame(closests[:recommendation_nb])

    return func.HttpResponse(str(selection['article_id'].tolist()))
