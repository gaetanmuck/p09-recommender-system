import flask
app = flask.Flask(__name__)
app.config
import requests, json
import pandas as pd, numpy as np

# Global variables
user_ids = pd.read_csv('./data/user_ids.csv').sample(100)['user_id'].tolist()
interests = pd.read_csv('./data/interests.csv')
clicks = pd.read_csv('./data/clicks.csv')
distances = np.load('./data/distances.npy')
articles_ids =  np.load('./data/article_ids.npy')
recommendation_nb = 5




@app.route('/', methods=['GET'])
def home():
    return flask.render_template('index.html', user_ids=user_ids)


@app.route('/recommendation', methods=['GET'])
def recommendation():

    user_id = flask.request.args.get('user_id')
    
    # Get user already read articles
    articles = clicks[clicks['user_id'] == int(user_id)]

    # Merge the weights
    articles = articles.merge(interests, on=['user_id', 'category_id'], how='left').rename(columns={'rate':'weight'})

    # We drop the categories that does not interests the user (rate == NaN)
    articles.dropna(inplace=True)

    # Take only usefull distances (only to save memory on Azure)
    dists = []
    indexes = []
    for ind in articles['article_id']:
        index = np.where(articles_ids == ind)[0][0]
        indexes.append(index)
        dists.append(distances[index])
    dists = pd.DataFrame(dists)
    dists['index'] = indexes



    # Preparing request params
    json_obj = {
        'articles': articles.to_json(),
        'distances': dists.to_json(),
        'articles_ids': articles_ids.tolist(),
        'recommendation_nb': recommendation_nb,
        'user_id': flask.request.args.get('user_id')
    }

    # Asking Azure service
    resp = requests.post('https://recommendation-system-gm.azurewebsites.net', json=json_obj)
    json_response = json.loads(resp.text)

    print(json_response)

    return json_response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)