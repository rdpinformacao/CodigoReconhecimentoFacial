import cv2

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorFisher.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(1)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30,30))


    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        cpf = ""
        cargo = ""

        if id == 1:
            nome = 'pessoa numero 1'
            cpf = '000.000.000-00'
            cargo = 'Tecnico'
            print('1' )
        elif id == 2:
            nome = 'Pessoa numero 2'
            cpf = '###.###.###-##'
            cargo = 'Analista'
            print('2' )
        elif (id != 1) and (id != 2):
            nome = 'Faca seu cadastro!'
            cpf = 'Nao consta na nossa base!'
            cargo = 'Sem funcao!'
            print('Nao existe!')

        cv2.putText(imagem, nome, (x, y + (a + 20)), font, 1, (255, 255, 255))
        cv2.putText(imagem, cpf, (x, y + (a + 40)), font, 1, (255, 255, 255))
        cv2.putText(imagem, cargo, (x, y + (a + 55)), font, 1, (255, 255, 255))
        #cv2.putText(imagem, str(confianca), (x, y + (a + 70)), font, 1, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()