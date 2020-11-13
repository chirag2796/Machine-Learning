import ktrain
from ktrain import text

IMDB_DATADIR = r"D:\Dev\Datasets\Text\Classification\Imdb\aclImdb"

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(IMDB_DATADIR,
                                                                       maxlen=500,
                                                                       preprocess_mode='bert',
                                                                       train_test_names=['train',
                                                                                         'test'],
                                                                       classes=['pos', 'neg'])
# print(x_train[0])

model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model,train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)

learner.fit_onecycle(2e-5, 1)