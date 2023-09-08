import cv2
import pickle
import extrair_gabarito as exg

# arquivo com os campos - posições X e Y de cada campo onde estão as respostas
campos = []
with open('campos.pkl', 'rb') as arquivo:
    campos = pickle.load(arquivo)

# arquivo com a questão correspondente ao campo
resp = []
with open('resp.pkl', 'rb') as arquivo:
    resp = pickle.load(arquivo)

respostas_corretas = ["1-A", "2-C", "3-B", "4-D", "5-A"]

video = cv2.VideoCapture(1)

while True:
    # lendo a imagem e fazendo um redimensionamento
    _, imagem = video.read()
    imagem = cv2.resize(imagem, (600, 700))

    # extraindo a imagem do gabarito e sua posição
    gabarito, bbox = exg.extrair_maior_ctn(imagem)
    img_gray = cv2.cvtColor(gabarito, cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.rectangle(imagem, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
    respostas = []

    # extraindo as posições de cada campo
    for id, vg in enumerate(campos):
        x = int(vg[0])
        y = int(vg[1])
        w = int(vg[2])
        h = int(vg[3])

        cv2.rectangle(gabarito, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img_th, (x, y), (x + w, y + h), (255, 255, 255), 1)

        campo = img_th[y:y + h, x:x + w]
        height, width = campo.shape[:2]
        tamanho = height + width
        pretos = cv2.countNonZero(campo)
        percentual = round((pretos / tamanho) * 100, 2)

        if percentual >= 15:
            cv2.rectangle(gabarito, (x, y), (x + w, y + h), (255, 0, 0), 2)
            respostas.append(resp[id])

    erros = 0
    acertos = 0

    if len(respostas) == len(respostas_corretas):
        for num, res in enumerate(respostas):
            if res == respostas_corretas[num]:
                acertos += 1
            else:
                erros += 1

        pontuacao = int(acertos * 6)
        cv2.putText(imagem, f'Acertos: {acertos}, pontos: {pontuacao}', (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    cv2.imshow('img', imagem)
    cv2.imshow('Gabarito', gabarito)
    cv2.imshow('IMG TH', img_th)
    cv2.waitKey(1)
