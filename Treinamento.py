import cv2
import os
import numpy as np



fisherface = cv2.face.FisherFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('Fotos_Capturadas', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int (os.path.split(caminhoImagem) [-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)

        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)

    return np.array(ids), faces

ids, faces = getImagemComId()
#print(faces)

print("Treinando...")

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

print("Treinamento realizado")
