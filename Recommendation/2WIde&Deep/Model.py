import tensorflow as tf


TRAIN_DATA_URL='file:///D:\PythonProject\Tensorflow2021_10_11\Recommendation\1EmbeddingMLP\data\modelSamples.csv'
samples_file_path=tf.keras.utils.get_file('modelSamples.csv',TRAIN_DATA_URL)

def get_dataset(file_path):
    dataset=tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value='?',
        num_epochs=1,
    )
    return dataset

# sample dataset size 110830/12(batch_size) = 9235
raw_samples_data=get_dataset(samples_file_path)

test_dataset=raw_samples_data.take(1000)
train_dataset=raw_samples_data.skip(1000) #tf.data.Dataset.skip() 方法用于创建一个数据集，该数据集从该数据集中跳过计数初始元素

#类别型特征处理
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary', 'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
GENRE_FEATURES = { 'userGenre1': genre_vocab,
                   'userGenre2': genre_vocab,
                   'userGenre3': genre_vocab,
                   'userGenre4': genre_vocab,
                   'userGenre5': genre_vocab,
                   'movieGenre1': genre_vocab,
                   'movieGenre2': genre_vocab,
                   'movieGenre3': genre_vocab}

categorical_columns=[]
for feature,vocab in GENRE_FEATURES.items(): #把用户设置的标签的列 做成onehot向量然后embedding10维度
    cat_col=tf.feature_column.categorical_column_with_vocabulary_list(key=feature,vocabulary_list=vocab)
    emb_col=tf.feature_column.embedding_column(cat_col,10)
    categorical_columns.append(emb_col)
#这里结束后应该就是8个类别的n*10 的矩阵 ，可能要拼起来
#https://blog.csdn.net/pearl8899/article/details/107946519
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)#表示我们movid最大去到1001,有1001种，如果movid为340，则在340处为1
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10) #取1003或者更大也可以，就是由于我们没有这么大的id ，那里都为0
categorical_columns.append(movie_emb_col)

user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)#同理用户id的最大值为30001
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
categorical_columns.append(user_emb_col)

#数值型特征的处理
#我们直接把特征值输入到 MLP 内，然后把特征逐个声明为 tf.feature_column.numeric_column
numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                   tf.feature_column.numeric_column('movieRatingCount'),
                     tf.feature_column.numeric_column('movieAvgRating'),
                     tf.feature_column.numeric_column('movieRatingStddev'),
                     tf.feature_column.numeric_column('userRatingCount'),
                     tf.feature_column.numeric_column('userAvgRating'),
                     tf.feature_column.numeric_column('userRatingStddev')]


preprocessing_layer = tf.keras.layers.DenseFeature(numerical_columns + categorical_columns)


model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


model.fit(train_dataset, epochs=10)


test_loss, test_accuracy = model.evaluate(test_dataset)


print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
