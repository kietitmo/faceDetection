from keras_facenet import FaceNet
import numpy as np
from sklearn.preprocessing import LabelEncoder

embedder = FaceNet()
encoder = LabelEncoder()

class Image_Embedder:
    def __init__(self):
        self.EMBEDDED_X = []
        self.transformed_y = []

    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')  # 3D(160x160x3)
        face_img = np.expand_dims(face_img, axis=0)

        # 4D (None x 160 x 160 x 3)
        face_img_embedded = embedder.embeddings(face_img)

        # 512D image (1 x 1 x 512)
        return face_img_embedded[0]

    def embed_trans(self, x, y):
        for image in x:
            self.EMBEDDED_X.append(self.get_embedding(image))

#         unknown_emb = np.ones(512)
#         self.EMBEDDED_X.append(unknown_emb)
#         y = y.tolist()
#         y.append('Unknown')

        encoder.fit(y)
        self.transformed_y = encoder.transform(y)
        return np.asarray(self.EMBEDDED_X), np.asarray(self.transformed_y)