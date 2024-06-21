import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Dense
import tensorflow as tf
import datetime

# Loading product data
df_prod = pd.read_csv('C:/002FRS/products.csv')
df_prod = pd.DataFrame(df_prod, columns=['product_id', 'product_name', 'description', 'item_day_id', 'from_time_1', 'to_time_1', 'from_time_2', 'to_time_2'])
assert 'product_name' in df_prod.columns, "product_name not found in df_prod columns"

# Loading item days data
df_days = pd.DataFrame(pd.read_excel("C:/002FRS/vb_item_days.xlsx"))
df_days = df_days.drop(['item_day_status'], axis='columns')
day_map = {"All": 0, "Fri": 3, "Sat": 4, "Sun": 5, "Mon": 6, "Tue": 7, "Wed": 8, "Thu": 9}
df_prod["item_day_id"] = df_prod["item_day_id"].replace(day_map).infer_objects(copy=False)
df_prod['description'] = df_prod['description'].fillna('')

# Text vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df_prod['description']).toarray()

# Time conversion
def time_to_float(time_str):
    if pd.isna(time_str):
        return 0.0
    if isinstance(time_str, float):
        return time_str
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60

df_prod['from_time_1'] = df_prod['from_time_1'].apply(time_to_float)
df_prod['to_time_1'] = df_prod['to_time_1'].apply(time_to_float)
df_prod['from_time_2'] = df_prod['from_time_2'].apply(time_to_float)
df_prod['to_time_2'] = df_prod['to_time_2'].apply(time_to_float)

# Standardizing time features
X_time = df_prod[['from_time_1', 'to_time_1', 'from_time_2', 'to_time_2']].values
scaler = StandardScaler()
X_time = scaler.fit_transform(X_time)
X_combined = np.hstack((X_text, X_time))

# Building Autoencoder model
input_dim = X_combined.shape[1]
encoding_dim = 128
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_combined, X_combined, epochs=100, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)

# Extracting encoded features
encoder_model = Model(inputs=input_layer, outputs=encoder)
encoded_features = encoder_model.predict(X_combined)


# Clustering
n_clusters = 19
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df_prod['cluster'] = kmeans.fit_predict(encoded_features)

# Loading user data
df_user = pd.read_csv('C:/002FRS/vb-users.csv')
df_user = pd.DataFrame(df_user, columns=['user_id', 'first_name', 'email'])

# Loading order data
df_list = pd.read_csv('C:/002FRS/vb_order_line.csv')
df_list = df_list.drop(['ingredient_amount', 'order_date', 'cancel_status', 'ingredients', 'cooking_instruction', 'order_line_date'], axis='columns')
df_order = pd.read_csv('C:/002FRS/vb_order_header.csv')
df_order = pd.DataFrame(df_order, columns=['order_id', 'order_number', 'user_id', 'branch_id'])
df_tot = df_order.groupby(['user_id', 'order_id']).size().groupby('user_id').size()
df_min = df_tot[df_tot >= 2].reset_index()[['user_id']]
df_regCust = df_order.merge(df_min, how='right', left_on='user_id', right_on='user_id')
df_all = df_list.merge(df_user.merge(df_order))

# User frequency
user_freq = df_all[['user_id', 'order_id']].groupby('user_id').count().reset_index()
merged_data = df_all.groupby('user_id').agg({
    'order_line_id': list,
    'order_id': list,
    'category_id': list,
    'product_id': list,
    'price': list,
    'quantity': list,
    'linetotal': list,
    'first_name': 'first',
    'email': 'first',
    'order_number': 'first',
    'branch_id': 'first'
}).reset_index()
pd.set_option('future.no_silent_downcasting', True)
# Creating user-product DataFrame
data = []
for index, row in merged_data.iterrows():
    user_id = row['user_id']
    product_id = row['product_id']
    data.append({'user_id': user_id, 'product_id': product_id})

user_prod_df = pd.DataFrame(data)

# Function to get three most frequent products
def three_most_frequent(products):
    if not products:
        return []
    count = Counter(products)
    most_common = count.most_common(3)
    return [item[0] for item in most_common]

user_prod_df['target'] = user_prod_df['product_id'].apply(three_most_frequent)
df = user_prod_df.copy()
df = df.explode('product_id')

# Mapping user and product IDs to indices
user_ids = df['user_id'].unique()
product_ids = df['product_id'].unique()
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
product_id_to_idx = {product_id: idx for idx, product_id in enumerate(product_ids)}
df['user_idx'] = df['user_id'].map(user_id_to_idx)
df['product_idx'] = df['product_id'].map(product_id_to_idx)
dataset = tf.data.Dataset.from_tensor_slices((df['user_idx'].values, df['product_idx'].values))



class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users, num_products, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_dim)
        self.dot = tf.keras.layers.Dot(axes=-1)
    
    def call(self, inputs):
        user_idx, product_idx = inputs
        user_embedding = self.user_embedding(user_idx)
        product_embedding = self.product_embedding(product_idx)
        dot_product = self.dot([user_embedding, product_embedding])
        return tf.reshape(dot_product, (-1, 1))

    def get_config(self):
        return {'num_users': self.num_users,
                'num_products': self.num_products,
                'embedding_dim': self.embedding_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# Preparing training data
train_dataset = tf.data.Dataset.from_tensor_slices(((df['user_idx'].values, df['product_idx'].values), tf.ones(len(df))))
train_dataset = train_dataset.shuffle(10000).batch(40)
train_dataset = train_dataset.map(lambda x, y: ((x[0], x[1]), y))

# Model training
num_users = len(user_ids)
num_products = len(product_ids)
embedding_dim = 32
model = RecommenderModel(num_users, num_products, embedding_dim)
model.compile(optimizer='adam', loss='binary_crossentropy')
tf.data.experimental.enable_debug_mode()
epochs = 10
model.fit(train_dataset, epochs=epochs, verbose=0)

# Function to get content-based recommendations
def get_content_based_recommendations(product_id, num_recommendations=8):
    product_index = df_prod.index[df_prod['product_id'] == product_id].tolist()[0]
    product_vector = X_combined[product_index]
    similarities = np.dot(X_combined, product_vector)
    similar_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]
    
    recommended_products = df_prod.iloc[similar_indices].copy()
    content_scores = similarities[similar_indices]
    
    recommended_products['content_score'] = content_scores
    return recommended_products, content_scores

# Function to get collaborative filtering recommendations
def get_collaborative_filtering_recommendations(user_id, num_recommendations=8):
    user_idx = user_id_to_idx.get(user_id, None)
    if user_idx is None:
        return pd.DataFrame(), []
    
    user_embedding = model.user_embedding(np.array([user_idx]))
    product_embeddings = model.product_embedding(np.arange(num_products))
    similarities = tf.tensordot(user_embedding, product_embeddings, axes=[-1, -1]).numpy().flatten()
    
    similar_indices = similarities.argsort()[-num_recommendations:][::-1]
    recommended_products = df_prod[df_prod['product_id'].isin(product_ids[similar_indices])].copy()
    collaborative_scores = similarities[similar_indices]
    
    recommended_products['collab_score'] = collaborative_scores
    return recommended_products, collaborative_scores

# Function to recommend products using hybrid approach
def recommend_products_hybrid(user_id, num_recommendations=8):
    try:
        # Getting user's frequently ordered products
        user_data = merged_data[merged_data['user_id'] == user_id]
        if user_data.empty:
            print("User data is empty.")
            return pd.DataFrame()

        user_product_ids = user_data['product_id'].values[0]
        if not user_product_ids:
            print("No user product IDs found.")
            return pd.DataFrame()

        # Taking the first product as the reference product
        product_id = user_product_ids[0]
        print(f"Reference product_id for user {user_id}: {product_id}")

        content_recommendations, content_scores = get_content_based_recommendations(product_id, num_recommendations)
        collaborative_recommendations, collaborative_scores = get_collaborative_filtering_recommendations(user_id, num_recommendations)

        combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates().reset_index(drop=True)

        current_day = datetime.datetime.now().strftime('%a')
        current_time = datetime.datetime.now().strftime('%H:%M')
        current_time_float = time_to_float(current_time)
        available_recommendations = combined_recommendations[
            ((combined_recommendations['item_day_id'] == 0) | (combined_recommendations['item_day_id'] == day_map[current_day])) &
            (((combined_recommendations['from_time_1'] <= current_time_float) & (current_time_float <= combined_recommendations['to_time_1'])) |
             ((combined_recommendations['from_time_2'] <= current_time_float) & (current_time_float <= combined_recommendations['to_time_2'])))
        ]

        if available_recommendations.empty:
            print("No products available at the current time.")
            return pd.DataFrame()

        available_recommendations['hybrid_score'] = available_recommendations.apply(
            lambda row: row.get('content_score', 0) + row.get('collab_score', 0), axis=1
        )

        available_recommendations = available_recommendations.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)
        return available_recommendations
    except Exception as e:
        print(f"Error in recommend_products_hybrid: {e}")
        return pd.DataFrame()

# Function to get most sold products
def get_most_sold_products(num_recommendations=8):
    product_sales = df_list['product_id'].value_counts().reset_index()
    product_sales.columns = ['product_id', 'sales_count']
    most_sold_products = product_sales.head(num_recommendations)
    return most_sold_products

# Function to recommend products for new users
def recommend_for_new_user(num_recommendations=8):
    most_sold_products = get_most_sold_products(num_recommendations)
    recommended_products = df_prod[df_prod['product_id'].isin(most_sold_products['product_id'])]
    recommended_products = recommended_products.merge(most_sold_products, on='product_id')
    recommended_products = recommended_products.sort_values(by='sales_count', ascending=False)
    return recommended_products

# Main recommendation function for users
def recommend_for_user(user_id, num_recommendations=8):
    try:
        if user_id not in user_ids:
            print("User is new. Recommending most sold products.")
            recommendations = recommend_for_new_user(num_recommendations)
            if recommendations is None or recommendations.empty:
                print("No recommendations available.")
                return pd.DataFrame()
            else:
                print(recommendations[['product_id', 'product_name', 'sales_count']])
                return recommendations
        else:
            print(f"Recommended products for user {user_id} at the current time:")
            recommendations = recommend_products_hybrid(user_id, num_recommendations)
            if recommendations is None or recommendations.empty:
                print("No recommendations found.")
                return pd.DataFrame()
            else:
                print(recommendations[['product_id', 'product_name']])
                return recommendations
    except Exception as e:
        print(f"Error in recommend_for_user: {e}")
        return pd.DataFrame()

'''silhouette_avg = silhouette_score(encoded_features, df_prod['cluster'])
print("Silhouette Score:", silhouette_avg)'''

import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model,f)