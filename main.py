from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_embedding.embedding import Image_Embedder
from data_embedding.faces_loading import ImagesLoader
from model_svm.svm_model import SVM_model
from recognize_realtime import Real_time_recognition

import numpy as np

if __name__ == "__main__":
    directory = "data_generation/faces"
#     faces_loaded = ImagesLoader(directory)
#     X, Y = faces_loaded.load_classes()
#
#     image_Embedder = Image_Embedder()
#     x_em, y_em = image_Embedder.embed_trans(X, Y)
#     np.savez_compressed('faces_embeddings_done_4classes_2.npz', x_em, y_em)

    faces_embeddings_done_4classes = np.load("faces_embeddings_done_4classes_2.npz")
    x_em = faces_embeddings_done_4classes['arr_0']
    y_em = faces_embeddings_done_4classes['arr_1']

    X_train, X_test, Y_train, Y_test = train_test_split(x_em, y_em, shuffle=True, random_state=17)

    model = SVM_model(X_train, Y_train)

    real_time = Real_time_recognition(model, directory)
    real_time.run()