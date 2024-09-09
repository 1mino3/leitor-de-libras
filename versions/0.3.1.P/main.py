import cv2
import mediapipe as mp
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class LeitorDeLibras(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurações da janela
        self.setWindowTitle('Leitor de Libras') # Título da janela
        self.setGeometry(100, 100, 800, 600) # Tamanho ao iniciar a janela
        self.setStyleSheet('background-color: #49d160;') # Background da janela

        # Layout principal
        self.layout = QVBoxLayout()

        # Título
        self.title_label = QLabel('Leitor de Libras', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet('font-size: 50px; font-family: "Times New Roman";')
        self.layout.addWidget(self.title_label)

        # Label de vídeo
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet('border: 2px solid black; ')
        self.video_label.setScaledContents(True)
        self.layout.addWidget(self.video_label)

        # Botão de Iniciar/Parar
        self.start_button = QPushButton('Iniciar', self)
        self.start_button.clicked.connect(self.toggle_video)
        self.start_button.setStyleSheet('padding: 20px; margin-top: 30px; background-color: #aff0a1; font-size: 20px; border-radius: 20px;')
        self.layout.addWidget(self.start_button)

        # Status
        self.status_label = QLabel('Clique em Iniciar para começar', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 15px')
        self.layout.addWidget(self.status_label)

        # Configurações do Mediapipe
        self.video = cv2.VideoCapture(0)
        self.hand = mp.solutions.hands
        self.Hand = self.hand.Hands(max_num_hands=1)
        self.mpDraw = mp.solutions.drawing_utils
        self.last_libra = None # para a interface da letra

        # Timer para atualizar o vídeo
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Container
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.is_recording = False # verifica se o button foi clicado

        # Pontos das mãos (para a detecção de movimento)
        self.point_0 = None
        self.point_1 = None
        self.point_2 = None
        self.point_3 = None
        self.point_4 = None
        self.point_5 = None
        self.point_6 = None
        self.point_7 = None
        self.point_8 = None
        self.point_9 = None
        self.point_10 = None
        self.point_11 = None
        self.point_12 = None
        self.point_13 = None
        self.point_14 = None
        self.point_15 = None
        self.point_16 = None
        self.point_17 = None
        self.point_18 = None
        self.point_19 = None
        self.point_20 = None

    def toggle_video(self):
        if not self.is_recording:
            self.timer.start(30)
            self.start_button.setText('Parar')
            self.is_recording = True
        else:
            self.timer.stop()
            self.start_button.setText('Iniciar')
            self.is_recording = False

    def update_frame(self):
        ''''

        Leia atentamente cada comentário antes de alterar algo. O código desta função é essencial, ou seja, um erro será fatal.

        Este código consiste em atualizar os frames com base no mediapipe. Sem ele, a função detectar_libra() não funcionará.

        OS COMENTÁRIOS ABAIXO SÃO PARA FINS DE DESENVOLVIMENTO E/OU TESTE.
        Os comandos em comentários, desenho dos pontos e desenho das conexões, são usado para mostrar os pontos e articulações, facilitando a detecção das condições. Caso esteja testando ou experimentando, não é necessária a utilização dos mesmos.
        No mediapipe, geralmente há a variável img. Neste código, será usada frame, que tem a mesma finalidade e parâmetro.
        x e y são para calcular a mão, assim como h e w.

        Caso tenha mais alguma observação necessária, será colocada aqui em futuras atualizações.

        '''


        ret, frame = self.video.read()
        if ret:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.Hand.process(frameRGB)
            h, w, _ = frame.shape

            pontos = [] # pontos: pontos[numero][eixo]
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for id, cord in enumerate(hand_landmarks.landmark):
                        cx, cy = int(cord.x * w), int(cord.y * h)
                        cz = cord.z
                        # cv2.putText(frameRGB, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (151, 15, 166), 2) # desenho dos pontos
                        pontos.append((cx, cy, cz))

                        if cx > x_max:
                            x_max = cx
                        if cx < x_min:
                            x_min = cx
                        if cy > y_max:
                            y_max = cy
                        if cy < y_min:
                            y_min = cy
                    cv2.rectangle(frameRGB, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                    # self.mpDraw.draw_landmarks(frameRGB, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)  # desenho das conexões entre os pontos (traços)


                libra = self.detectar_libra(pontos)
                if libra:
                    self.last_libra = libra
                    self.status_label.setText(f"Letra detectada: {libra}")
                    if self.last_libra  not in "HJK":
                        cv2.putText(frameRGB, libra, (x_max + 30, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                    else:
                        cv2.putText(frameRGB, self.last_libra, (x_max + 30, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

            # Atualiza o vídeo na interface
            qimg = QImage(frameRGB.data, frameRGB.shape[1], frameRGB.shape[0], frameRGB.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

    def detectar_libra(self, pontos):
        '''

        Leia atentamente cada comentário antes de alterar qualquer código.

        Este código consiste em detectar a letra. Para funcionar, a função update_frame() é necessária.

        OS COMENTÁRIOS ABAIXO SÃO PARA FINS DE DESENVOLVIMENTO E/OU TESTE.
        :param pontos: pontos são as cordenadas da mão, onde tirando as variáveis mpDraw e a dos textos das tags, é possível localizá-las. O eixo
        X é escrito por pontos[p][0], o Y por [p][1] e Z por [p][2].
        Se, no eixo X, [p1][0] > [p2][0] significa que p1 está à direita de p2;
        Se, no eixo Y, [p1][1] > [p2][1] significa que p1 está abaixo de p2;
        Se, no eixo Z, [p1][2] > [p2][2] significa que p1 está atrás de p2.

        Para ver o código de uma letra ou libra, pesquise "Letra [letra em maiúsculo]" ou "Libra [expressão ou palavra com a inicial em maiúsculo]" com a devida formartação das letras em uma barra de pesquisa de texto. Algumas libras como "Eu te amo" podem ser encontradas por emojis.
        Algunas letras, como K e J, não estão em ordem alfabética.
        Qualquer conhecimento de algum erro neste código, informe ao proprietário deste programa.
        Utilize "if " para cada condição, pois "elif" pode dar conflito com as condições de movimento.
        Os parâmetros point servem para a detecção de movimento nas libras. Evite criar ou alterá-los no código, já que danificará a detecção de algumas.
        Os pontos condicionais em comentários são os que "teoricamente" devem funcionar, mas por bug do próprio mediapipe não dão certo. Até que o mediapipe corrija tais, não os tire de comentários.
        Letra T desativada. Por erro do mediapipe, a articulação da libra está sendo detectada de forma errada. Até que o mediapipe hands seja aprimorado ou seja descoberta uma nova forma de diferenciar F e T, continuará assim😞

        :return: Retorna a libra detectada. Os point são alterados

        Caso tenha mais alguma observação necessária, será colocada aqui em futuras atualizações.

        '''

        pass
        if len(pontos) > 20:


            if (  #  Letra A
                pontos[8][1] > pontos[5][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[20][1] > pontos[17][1] and
                pontos[4][1] < pontos[5][1] and

                pontos[4][0] > pontos[6][0]
            ):
                return "A"


            if (  #  Letra B
                pontos[8][1] < pontos[7][1] and
                pontos[12][1] < pontos[11][1] and
                pontos[16][1] < pontos[15][1] and
                pontos[20][1] < pontos[19][1] and
                pontos[4][0] < pontos[1][0] and
                pontos[4][2] < pontos[13][2]
            ):
                return "B"


            if  (  #  Letra C
                pontos[8][2] > pontos[12][2] and
                pontos[12][2] > pontos[16][2] and
                pontos[16][2] > pontos[20][2] and
                pontos[4][0] > pontos[3][0] and
                pontos[4][2] > pontos[20][2] and
                pontos[4][1] > pontos[8][1] and
                pontos[17][2] < pontos[5][2] and
                pontos[17][2] < pontos[13][2] and
                pontos[17][2] < pontos[9][2] and
                pontos[17][2] < pontos[4][2] and
                pontos[4][1] > pontos[8][1] and
                pontos[4][1] > pontos[16][1] and
                pontos[4][1] > pontos[18][1] and
                pontos[4][1] > pontos[19][1] and
                pontos[4][1] > pontos[20][1] and
                pontos[4][1] > pontos[14][1] and
                pontos[4][1] > pontos[15][1] and
                pontos[4][1] > pontos[16][1] and
                pontos[4][1] > pontos[10][1] and
                pontos[4][1] > pontos[11][1] and
                pontos[4][1] > pontos[12][1] and
                pontos[4][1] > pontos[6][1] and
                pontos[4][1] > pontos[7][1] and
                pontos[4][1] > pontos[8][1] and
                pontos[20][1] < pontos[18][1]
            ):
                return "C"


            if (  #  Letra D
                pontos[8][1] < pontos[10][1] and
                pontos[12][1] > pontos[10][1] and
                pontos[16][1] > pontos[14][1] and
                pontos[20][1] > pontos[18][1] and
                pontos[4][1] < pontos[16][1] and
                pontos[4][1] > pontos[10][1] and
                pontos[4][1] < pontos[3][1] and
                pontos[4][1] < pontos[2][1] and
                pontos[4][0] < pontos[3][0] and
                pontos[12][1] < pontos[3][1] and
                pontos[11][1] < pontos[3][1] and
                pontos[11][1] < pontos[9][1]
            ):
                return "D"


            if (  # Letra E
                pontos[8][1] < pontos[5][1] and
                pontos[20][1] > pontos[19][1] and
                pontos[16][1] > pontos[15][1] and
                pontos[12][1] > pontos[11][1] and
                pontos[4][1] > pontos[5][1] and
                pontos[4][2] < pontos[2][2] and
                pontos[4][0] < pontos[3][0] and
                pontos[8][1] > pontos[7][1] and
                pontos[20][1] > pontos[19][1] and
                pontos[16][1] > pontos[15][1] and
                pontos[12][1] > pontos[11][1]
            ):
                return "E"


            if ( # Letra F
                # pontos[4][2] > pontos[0][2] and
                # pontos[4][2] > pontos[1][2] and
                # pontos[4][2] > pontos[2][2] and
                # pontos[4][2] > pontos[3][2] and
                # pontos[4][2] > pontos[7][2] and
                pontos[6][1] > pontos[10][1] and
                pontos[6][1] > pontos[14][1] and
                pontos[6][1] > pontos[19][1] and
                pontos[7][1] > pontos[10][1] and
                pontos[7][1] > pontos[14][1] and
                pontos[7][1] > pontos[19][1] and
                pontos[8][1] > pontos[10][1] and
                pontos[8][1] > pontos[14][1] and
                pontos[8][1] > pontos[19][1] and
                pontos[20][1] > pontos[15][1]
            ):
                return "F"


            if ( # Letra G
                pontos[8][1] < pontos[0][1] and
                pontos[8][1] < pontos[1][1] and
                pontos[8][1] < pontos[2][1] and
                pontos[8][1] < pontos[3][1] and
                pontos[8][1] < pontos[5][1] and
                pontos[8][1] < pontos[6][1] and
                pontos[8][1] < pontos[7][1] and
                pontos[8][1] < pontos[9][1] and
                pontos[8][1] < pontos[10][1] and
                pontos[8][1] < pontos[11][1] and
                pontos[8][1] < pontos[12][1] and
                pontos[8][1] < pontos[13][1] and
                pontos[8][1] < pontos[14][1] and
                pontos[8][1] < pontos[15][1] and
                pontos[8][1] < pontos[16][1] and
                pontos[8][1] < pontos[17][1] and
                pontos[8][1] < pontos[18][1] and
                pontos[8][1] < pontos[19][1] and
                pontos[8][1] < pontos[20][1] and

                pontos[7][1] < pontos[0][1] and
                pontos[7][1] < pontos[1][1] and
                pontos[7][1] < pontos[2][1] and
                pontos[7][1] < pontos[3][1] and
                pontos[7][1] < pontos[5][1] and
                pontos[7][1] < pontos[6][1] and
                pontos[7][1] > pontos[8][1] and
                pontos[7][1] < pontos[9][1] and
                pontos[7][1] < pontos[10][1] and
                pontos[7][1] < pontos[11][1] and
                pontos[7][1] < pontos[12][1] and
                pontos[7][1] < pontos[13][1] and
                pontos[7][1] < pontos[14][1] and
                pontos[7][1] < pontos[15][1] and
                pontos[7][1] < pontos[16][1] and
                pontos[7][1] < pontos[17][1] and
                pontos[7][1] < pontos[18][1] and
                pontos[7][1] < pontos[19][1] and
                pontos[7][1] < pontos[20][1] and

                pontos[6][1] < pontos[0][1] and
                pontos[6][1] < pontos[1][1] and
                pontos[6][1] < pontos[2][1] and
                pontos[6][1] < pontos[3][1] and
                pontos[6][1] < pontos[5][1] and
                pontos[6][1] > pontos[7][1] and
                pontos[6][1] > pontos[8][1] and
                pontos[6][1] < pontos[9][1] and
                pontos[6][1] < pontos[10][1] and
                pontos[6][1] < pontos[11][1] and
                pontos[6][1] < pontos[12][1] and
                pontos[6][1] < pontos[13][1] and
                pontos[6][1] < pontos[14][1] and
                pontos[6][1] < pontos[15][1] and
                pontos[6][1] < pontos[16][1] and
                pontos[6][1] < pontos[17][1] and
                pontos[6][1] < pontos[18][1] and
                pontos[6][1] < pontos[19][1] and
                pontos[6][1] < pontos[20][1] and

                pontos[4][1] < pontos[1][1] and
                pontos[4][1] < pontos[2][1] and
                pontos[4][1] < pontos[3][1] and
                pontos[4][1] < pontos[5][1] and
                pontos[4][1] > pontos[6][1] and
                pontos[4][1] > pontos[7][1] and
                pontos[4][1] > pontos[8][1] and
                pontos[4][1] < pontos[9][1] and
                pontos[4][1] < pontos[10][1] and
                pontos[4][1] < pontos[11][1] and
                pontos[4][1] < pontos[12][1] and
                pontos[4][1] < pontos[13][1] and
                pontos[4][1] < pontos[14][1] and
                pontos[4][1] < pontos[15][1] and
                pontos[4][1] < pontos[16][1] and
                pontos[4][1] < pontos[17][1] and
                pontos[4][1] < pontos[18][1] and
                pontos[4][1] < pontos[19][1] and
                pontos[4][1] < pontos[20][1] and
                pontos[4][2] < pontos[6][2] and
                pontos[4][0] > pontos[6][0] and

                pontos[20][1] > pontos[17][1] and
                pontos[20][1] > pontos[13][1] and
                pontos[20][1] > pontos[9][1] and
                pontos[20][1] > pontos[5][1] and
                pontos[16][1] > pontos[17][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[16][1] > pontos[9][1] and
                pontos[16][1] > pontos[5][1] and
                pontos[12][1] > pontos[17][1] and
                pontos[12][1] > pontos[13][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[12][1] > pontos[5][1] and
                pontos[19][1] > pontos[17][1] and
                pontos[19][1] > pontos[13][1] and
                pontos[19][1] > pontos[9][1] and
                pontos[19][1] > pontos[5][1] and
                pontos[15][1] > pontos[17][1] and
                pontos[15][1] > pontos[13][1] and
                pontos[15][1] > pontos[9][1] and
                pontos[15][1] > pontos[5][1] and
                pontos[11][1] > pontos[17][1] and
                pontos[11][1] > pontos[13][1] and
                pontos[11][1] > pontos[9][1] and
                pontos[11][1] > pontos[5][1] and

                pontos[11][1] > pontos[14][1] and
                pontos[12][1] > pontos[14][1] and
                pontos[11][0] < pontos[4][0]

            ):
                return "G"


            if (  # Letra I

                # Todos os pontos abaixo, com exceção de 14, 10, 6, 20, 4
                pontos[20][1] < pontos[0][1] and
                pontos[20][1] < pontos[1][1] and
                pontos[20][1] < pontos[2][1] and
                pontos[20][1] < pontos[3][1] and
                pontos[20][1] < pontos[5][1] and
                pontos[20][1] < pontos[7][1] and
                pontos[20][1] < pontos[8][1] and
                pontos[20][1] < pontos[9][1] and
                pontos[20][1] < pontos[11][1] and
                pontos[20][1] < pontos[12][1] and
                pontos[20][1] < pontos[13][1] and
                pontos[20][1] < pontos[15][1] and
                pontos[20][1] < pontos[16][1] and
                pontos[20][1] < pontos[17][1] and
                pontos[20][1] < pontos[18][1] and
                pontos[20][1] < pontos[19][1] and

                # Todos os pontos abaixo, com exceção de 14, 10, 6, 20, 19, 4
                pontos[19][1] < pontos[0][1] and
                pontos[19][1] < pontos[1][1] and
                pontos[19][1] < pontos[2][1] and
                pontos[19][1] < pontos[3][1] and
                pontos[19][1] < pontos[5][1] and
                pontos[19][1] < pontos[7][1] and
                pontos[19][1] < pontos[8][1] and
                pontos[19][1] < pontos[9][1] and
                pontos[19][1] < pontos[11][1] and
                pontos[19][1] < pontos[12][1] and
                pontos[19][1] < pontos[13][1] and
                pontos[19][1] < pontos[15][1] and
                pontos[19][1] < pontos[16][1] and
                pontos[19][1] < pontos[17][1] and
                pontos[19][1] < pontos[18][1] and

                # Todos os pontos abaixo, com exceção de 14, 10, 6, 20, 19, 18, 4
                pontos[18][1] < pontos[0][1] and
                pontos[18][1] < pontos[1][1] and
                pontos[18][1] < pontos[2][1] and
                pontos[18][1] < pontos[3][1] and
                pontos[18][1] < pontos[5][1] and
                pontos[18][1] < pontos[7][1] and
                pontos[18][1] < pontos[8][1] and
                pontos[18][1] < pontos[9][1] and
                pontos[18][1] < pontos[11][1] and
                pontos[18][1] < pontos[12][1] and
                pontos[18][1] < pontos[13][1] and
                pontos[18][1] < pontos[15][1] and
                pontos[18][1] < pontos[16][1] and
                pontos[18][1] < pontos[17][1] and

                pontos[4][0] < pontos[3][0]
            ):
                return "I"

            if (  # Letra H
                self.point_8 != None and
                self.point_12 != None
            ):
                X_move_8 = pontos[8][0] - self.point_8[0]
                X_move_12 = pontos[12][0] - self.point_12[0]
                if (
                    X_move_8 > 5 and
                    X_move_12 > 5 and
                    pontos[8][1] < pontos[7][1] and
                    pontos[12][1] < pontos[11][1]
                ):
                    # Redefinição dos pontos das mãos (para a detecção de movimento)
                    self.point_0 = None
                    self.point_1 = None
                    self.point_2 = None
                    self.point_3 = None
                    self.point_4 = None
                    self.point_5 = None
                    self.point_6 = None
                    self.point_7 = None
                    self.point_8 = None
                    self.point_9 = None
                    self.point_10 = None
                    self.point_11 = None
                    self.point_12 = None
                    self.point_13 = None
                    self.point_14 = None
                    self.point_15 = None
                    self.point_16 = None
                    self.point_17 = None
                    self.point_18 = None
                    self.point_19 = None
                    self.point_20 = None
                    if self.last_libra != "H":
                        self.last_libra = "H"
                        return "H"
                    else:
                        return "H"



            if ( # Letra J
                self.point_8 != None and
                self.point_12 != None and
                self.point_0 != None
            ):
                X_move_8 = pontos[8][0] - self.point_8[0]
                X_move_12 = pontos[12][0] - self.point_12[0]
                Y_move_8 = pontos[8][1] - self.point_8[1]
                Y_move_12 = pontos[12][1] - self.point_12[1]

                if (
                        X_move_8 > 5 and
                        X_move_12 > 5 and
                        Y_move_8 > 5 and
                        Y_move_12 > 5 and
                        pontos[8][1] > pontos[7][1] and  # Dedo mindinho mais baixo que a base da palma
                        pontos[12][1] > pontos[11][1] and  # Dedo indicador mais baixo que a base da palma
                        pontos[0][1] > pontos[1][1] and  # Palma voltada para baixo
                        pontos[0][1] > pontos[2][1]
                ):
                    # Redefinição dos pontos das mãos (para a detecção de movimento)
                    self.point_0 = None
                    self.point_1 = None
                    self.point_2 = None
                    self.point_3 = None
                    self.point_4 = None
                    self.point_5 = None
                    self.point_6 = None
                    self.point_7 = None
                    self.point_8 = None
                    self.point_9 = None
                    self.point_10 = None
                    self.point_11 = None
                    self.point_12 = None
                    self.point_13 = None
                    self.point_14 = None
                    self.point_15 = None
                    self.point_16 = None
                    self.point_17 = None
                    self.point_18 = None
                    self.point_19 = None
                    self.point_20 = None
                    if self.last_libra != "J":
                        self.last_libra = "J"
                        return "J"
                    else:
                        return "J"


            if (  # Letra K
                    self.point_8 != None and
                    self.point_12 != None
            ):
                X_move_8 = pontos[8][0] - self.point_8[0]
                X_move_12 = pontos[12][0] - self.point_12[0]
                if (
                        X_move_8 < -5 and
                        X_move_12 < -5 and
                        pontos[8][1] < pontos[7][1] and
                        pontos[12][1] < pontos[11][1]
                ):
                    # Redefinição dos pontos das mãos (para a detecção de movimento)
                    self.point_0 = None
                    self.point_1 = None
                    self.point_2 = None
                    self.point_3 = None
                    self.point_4 = None
                    self.point_5 = None
                    self.point_6 = None
                    self.point_7 = None
                    self.point_8 = None
                    self.point_9 = None
                    self.point_10 = None
                    self.point_11 = None
                    self.point_12 = None
                    self.point_13 = None
                    self.point_14 = None
                    self.point_15 = None
                    self.point_16 = None
                    self.point_17 = None
                    self.point_18 = None
                    self.point_19 = None
                    self.point_20 = None
                    if self.last_libra != "K":
                        self.last_libra = "K"
                        return "K"
                    else:
                        return "K"


            if ( # Letra L
                pontos[8][1] < pontos[0][1] and
                pontos[8][1] < pontos[1][1] and
                pontos[8][1] < pontos[2][1] and
                pontos[8][1] < pontos[3][1] and
                pontos[8][1] < pontos[4][1] and
                pontos[8][1] < pontos[5][1] and
                pontos[8][1] < pontos[6][1] and
                pontos[8][1] < pontos[7][1] and
                pontos[8][1] < pontos[9][1] and
                pontos[8][1] < pontos[10][1] and
                pontos[8][1] < pontos[11][1] and
                pontos[8][1] < pontos[12][1] and
                pontos[8][1] < pontos[13][1] and
                pontos[8][1] < pontos[14][1] and
                pontos[8][1] < pontos[15][1] and
                pontos[8][1] < pontos[16][1] and
                pontos[8][1] < pontos[17][1] and
                pontos[8][1] < pontos[18][1] and
                pontos[8][1] < pontos[19][1] and
                pontos[8][1] < pontos[20][1] and

                pontos[7][1] < pontos[0][1] and
                pontos[7][1] < pontos[1][1] and
                pontos[7][1] < pontos[2][1] and
                pontos[7][1] < pontos[3][1] and
                pontos[7][1] < pontos[4][1] and
                pontos[7][1] < pontos[5][1] and
                pontos[7][1] < pontos[6][1] and
                pontos[7][1] > pontos[8][1] and
                pontos[7][1] < pontos[9][1] and
                pontos[7][1] < pontos[10][1] and
                pontos[7][1] < pontos[11][1] and
                pontos[7][1] < pontos[12][1] and
                pontos[7][1] < pontos[13][1] and
                pontos[7][1] < pontos[14][1] and
                pontos[7][1] < pontos[15][1] and
                pontos[7][1] < pontos[16][1] and
                pontos[7][1] < pontos[17][1] and
                pontos[7][1] < pontos[18][1] and
                pontos[7][1] < pontos[19][1] and
                pontos[7][1] < pontos[20][1] and

                pontos[6][1] < pontos[0][1] and
                pontos[6][1] < pontos[1][1] and
                pontos[6][1] < pontos[2][1] and
                pontos[6][1] < pontos[3][1] and
                pontos[6][1] < pontos[4][1] and
                pontos[6][1] < pontos[5][1] and
                pontos[6][1] > pontos[7][1] and
                pontos[6][1] > pontos[8][1] and
                pontos[6][1] < pontos[9][1] and
                pontos[6][1] < pontos[10][1] and
                pontos[6][1] < pontos[11][1] and
                pontos[6][1] < pontos[12][1] and
                pontos[6][1] < pontos[13][1] and
                pontos[6][1] < pontos[14][1] and
                pontos[6][1] < pontos[15][1] and
                pontos[6][1] < pontos[16][1] and
                pontos[6][1] < pontos[17][1] and
                pontos[6][1] < pontos[18][1] and
                pontos[6][1] < pontos[19][1] and
                pontos[6][1] < pontos[20][1] and

                pontos[4][0] > pontos[0][0] and
                pontos[4][0] > pontos[1][0] and
                pontos[4][0] > pontos[2][0] and
                pontos[4][0] > pontos[3][0] and
                pontos[4][0] > pontos[5][0] and
                pontos[4][0] > pontos[7][0] and
                pontos[4][0] > pontos[8][0] and
                pontos[4][0] > pontos[9][0] and
                pontos[4][0] > pontos[10][0] and
                pontos[4][0] > pontos[11][0] and
                pontos[4][0] > pontos[12][0] and
                pontos[4][0] > pontos[13][0] and
                pontos[4][0] > pontos[14][0] and
                pontos[4][0] > pontos[15][0] and
                pontos[4][0] > pontos[16][0] and
                pontos[4][0] > pontos[17][0] and
                pontos[4][0] > pontos[18][0] and
                pontos[4][0] > pontos[19][0] and
                pontos[4][0] > pontos[20][0] and

                pontos[3][0] > pontos[0][0] and
                pontos[3][0] > pontos[1][0] and
                pontos[3][0] > pontos[2][0] and
                pontos[3][0] > pontos[5][0] and
                pontos[3][0] > pontos[9][0] and
                pontos[3][0] > pontos[10][0] and
                pontos[3][0] > pontos[11][0] and
                pontos[3][0] > pontos[12][0] and
                pontos[3][0] > pontos[13][0] and
                pontos[3][0] > pontos[14][0] and
                pontos[3][0] > pontos[15][0] and
                pontos[3][0] > pontos[16][0] and
                pontos[3][0] > pontos[17][0] and
                pontos[3][0] > pontos[18][0] and
                pontos[3][0] > pontos[19][0] and
                pontos[3][0] > pontos[20][0]


            ):
                return "L"


            if ( # Letra M
                pontos[8][1] > pontos[0][1] and
                pontos[8][1] > pontos[1][1] and
                pontos[8][1] > pontos[2][1] and
                pontos[8][1] > pontos[3][1] and
                pontos[8][1] > pontos[4][1] and
                pontos[8][1] > pontos[5][1] and
                pontos[8][1] > pontos[6][1] and
                pontos[8][1] > pontos[7][1] and
                pontos[8][1] > pontos[9][1] and
                pontos[8][1] > pontos[10][1] and
                pontos[8][1] > pontos[11][1] and
                pontos[8][1] > pontos[13][1] and
                pontos[8][1] > pontos[14][1] and
                pontos[8][1] > pontos[15][1] and
                pontos[8][1] > pontos[17][1] and
                pontos[8][1] > pontos[18][1] and
                pontos[8][1] > pontos[19][1] and
                pontos[8][1] > pontos[20][1] and

                pontos[12][1] > pontos[0][1] and
                pontos[12][1] > pontos[1][1] and
                pontos[12][1] > pontos[2][1] and
                pontos[12][1] > pontos[3][1] and
                pontos[12][1] > pontos[4][1] and
                pontos[12][1] > pontos[5][1] and
                pontos[12][1] > pontos[6][1] and
                pontos[12][1] > pontos[7][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[12][1] > pontos[10][1] and
                pontos[12][1] > pontos[11][1] and
                pontos[12][1] > pontos[13][1] and
                pontos[12][1] > pontos[14][1] and
                pontos[12][1] > pontos[15][1] and
                pontos[12][1] > pontos[17][1] and
                pontos[12][1] > pontos[18][1] and
                pontos[12][1] > pontos[19][1] and
                pontos[12][1] > pontos[20][1] and

                pontos[16][1] > pontos[0][1] and
                pontos[16][1] > pontos[1][1] and
                pontos[16][1] > pontos[2][1] and
                pontos[16][1] > pontos[3][1] and
                pontos[16][1] > pontos[4][1] and
                pontos[16][1] > pontos[5][1] and
                pontos[16][1] > pontos[6][1] and
                pontos[16][1] > pontos[7][1] and
                pontos[16][1] > pontos[9][1] and
                pontos[16][1] > pontos[10][1] and
                pontos[16][1] > pontos[11][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[16][1] > pontos[14][1] and
                pontos[16][1] > pontos[15][1] and
                pontos[16][1] > pontos[17][1] and
                pontos[16][1] > pontos[18][1] and
                pontos[16][1] > pontos[19][1] and
                pontos[16][1] > pontos[20][1] and

                pontos[0][1] < pontos[1][1] and
                pontos[0][1] < pontos[2][1] and
                pontos[0][1] < pontos[3][1] and
                pontos[0][1] < pontos[4][1] and
                pontos[0][1] < pontos[5][1] and
                pontos[0][1] < pontos[6][1] and
                pontos[0][1] < pontos[7][1] and
                pontos[0][1] < pontos[8][1] and
                pontos[0][1] < pontos[9][1] and
                pontos[0][1] < pontos[10][1] and
                pontos[0][1] < pontos[11][1] and
                pontos[0][1] < pontos[12][1] and
                pontos[0][1] < pontos[13][1] and
                pontos[0][1] < pontos[14][1] and
                pontos[0][1] < pontos[15][1] and
                pontos[0][1] < pontos[16][1] and
                pontos[0][1] < pontos[17][1] and
                pontos[0][1] < pontos[18][1] and
                pontos[0][1] < pontos[19][1] and
                pontos[0][1] < pontos[20][1]

            ):
                return "M"


            if ( # Letra N
                pontos[8][1] > pontos[0][1] and
                pontos[8][1] > pontos[1][1] and
                pontos[8][1] > pontos[2][1] and
                pontos[8][1] > pontos[3][1] and
                pontos[8][1] > pontos[4][1] and
                pontos[8][1] > pontos[5][1] and
                pontos[8][1] > pontos[6][1] and
                pontos[8][1] > pontos[7][1] and
                pontos[8][1] > pontos[9][1] and
                pontos[8][1] > pontos[10][1] and
                pontos[8][1] > pontos[11][1] and
                pontos[8][1] > pontos[13][1] and
                pontos[8][1] > pontos[14][1] and
                pontos[8][1] > pontos[15][1] and
                pontos[8][1] > pontos[17][1] and
                pontos[8][1] > pontos[18][1] and
                pontos[8][1] > pontos[19][1] and
                pontos[8][1] > pontos[20][1] and

                pontos[12][1] > pontos[0][1] and
                pontos[12][1] > pontos[1][1] and
                pontos[12][1] > pontos[2][1] and
                pontos[12][1] > pontos[3][1] and
                pontos[12][1] > pontos[4][1] and
                pontos[12][1] > pontos[5][1] and
                pontos[12][1] > pontos[6][1] and
                pontos[12][1] > pontos[7][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[12][1] > pontos[10][1] and
                pontos[12][1] > pontos[11][1] and
                pontos[12][1] > pontos[13][1] and
                pontos[12][1] > pontos[14][1] and
                pontos[12][1] > pontos[15][1] and
                pontos[12][1] > pontos[17][1] and
                pontos[12][1] > pontos[18][1] and
                pontos[12][1] > pontos[19][1] and
                pontos[12][1] > pontos[20][1] and

                pontos[16][1] < pontos[12][1] and
                pontos[16][1] < pontos[11][1] and
                pontos[16][1] < pontos[8][1] and
                pontos[16][1] < pontos[7][1] and

                pontos[0][1] < pontos[1][1] and
                pontos[0][1] < pontos[2][1] and
                pontos[0][1] < pontos[3][1] and
                pontos[0][1] < pontos[4][1] and
                pontos[0][1] < pontos[5][1] and
                pontos[0][1] < pontos[6][1] and
                pontos[0][1] < pontos[7][1] and
                pontos[0][1] < pontos[8][1] and
                pontos[0][1] < pontos[9][1] and
                pontos[0][1] < pontos[10][1] and
                pontos[0][1] < pontos[11][1] and
                pontos[0][1] < pontos[12][1] and
                pontos[0][1] < pontos[13][1] and
                pontos[0][1] < pontos[14][1] and
                pontos[0][1] < pontos[15][1] and
                pontos[0][1] < pontos[16][1] and
                pontos[0][1] < pontos[17][1] and
                pontos[0][1] < pontos[18][1] and
                pontos[0][1] < pontos[19][1] and
                pontos[0][1] < pontos[20][1]
            ):
                return "N"


            if ( # Letra O
                pontos[8][2] > pontos[12][2] and
                pontos[12][2] > pontos[16][2] and
                pontos[16][2] > pontos[20][2] and
                pontos[17][2] < pontos[5][2] and
                pontos[17][2] < pontos[13][2] and
                pontos[17][2] < pontos[9][2] and
                pontos[4][1] < pontos[12][1] # and # Se a condição a seguir estiver ativa, tirar o comentário do "and"
                # pontos[20][1] < pontos[13][1] # Condição com possível problema. Em caso de erros em outras mãos, colocar sob comentário.
            ):
                return "O"


            if ( # Letra P
                pontos[8][1] < pontos[0][1] and
                pontos[8][1] < pontos[1][1] and
                pontos[8][1] < pontos[2][1] and
                pontos[8][1] < pontos[3][1] and
                pontos[8][1] < pontos[4][1] and
                pontos[8][1] < pontos[5][1] and
                pontos[8][1] < pontos[6][1] and
                pontos[8][1] < pontos[7][1] and
                pontos[8][1] < pontos[9][1] and
                pontos[8][1] < pontos[10][1] and
                pontos[8][1] < pontos[11][1] and
                pontos[8][1] < pontos[12][1] and
                pontos[8][1] < pontos[13][1] and
                pontos[8][1] < pontos[14][1] and
                pontos[8][1] < pontos[15][1] and
                pontos[8][1] < pontos[16][1] and
                pontos[8][1] < pontos[17][1] and
                pontos[8][1] < pontos[18][1] and
                pontos[8][1] < pontos[19][1] and
                pontos[8][1] < pontos[20][1] and

                pontos[7][1] < pontos[0][1] and
                pontos[7][1] < pontos[1][1] and
                pontos[7][1] < pontos[2][1] and
                pontos[7][1] < pontos[3][1] and
                pontos[7][1] < pontos[4][1] and
                pontos[7][1] < pontos[5][1] and
                pontos[7][1] < pontos[6][1] and
                pontos[7][1] < pontos[9][1] and
                pontos[7][1] < pontos[10][1] and
                pontos[7][1] < pontos[11][1] and
                pontos[7][1] < pontos[12][1] and
                pontos[7][1] < pontos[13][1] and
                pontos[7][1] < pontos[14][1] and
                pontos[7][1] < pontos[15][1] and
                pontos[7][1] < pontos[16][1] and
                pontos[7][1] < pontos[17][1] and
                pontos[7][1] < pontos[18][1] and
                pontos[7][1] < pontos[19][1] and
                pontos[7][1] < pontos[20][1] and

                pontos[4][1] < pontos[11][1] and
                pontos[4][1] < pontos[12][1] and

                pontos[12][0] > pontos[0][0] and
                pontos[12][0] > pontos[1][0] and
                pontos[12][0] > pontos[2][0] and
                pontos[12][0] > pontos[3][0] and
                pontos[12][0] > pontos[4][0] and
                pontos[12][0] > pontos[5][0] and
                pontos[12][0] > pontos[6][0] and
                pontos[12][0] > pontos[7][0] and
                pontos[12][0] > pontos[9][0] and
                pontos[12][0] > pontos[10][0] and
                pontos[12][0] > pontos[11][0] and
                pontos[12][0] > pontos[13][0] and
                pontos[12][0] > pontos[14][0] and
                pontos[12][0] > pontos[15][0] and
                pontos[12][0] > pontos[16][0] and
                pontos[12][0] > pontos[17][0] and
                pontos[12][0] > pontos[18][0] and
                pontos[12][0] > pontos[19][0] and
                pontos[12][0] > pontos[20][0] and
                pontos[12][1] < pontos[15][1] and

                pontos[11][0] > pontos[0][0] and
                pontos[11][0] > pontos[1][0] and
                pontos[11][0] > pontos[2][0] and
                pontos[11][0] > pontos[3][0] and
                pontos[11][0] > pontos[4][0] and
                pontos[11][0] > pontos[5][0] and
                pontos[11][0] > pontos[6][0] and
                pontos[11][0] > pontos[7][0] and
                pontos[11][0] > pontos[9][0] and
                pontos[11][0] > pontos[10][0] and
                pontos[11][0] > pontos[13][0] and
                pontos[11][0] > pontos[14][0] and
                pontos[11][0] > pontos[15][0] and
                pontos[11][0] > pontos[16][0] and
                pontos[11][0] > pontos[17][0] and
                pontos[11][0] > pontos[18][0] and
                pontos[11][0] > pontos[19][0] and
                pontos[11][0] > pontos[20][0]
            ):
                return "P"


            if ( # Letra Q
                pontos[0][1] < pontos[1][1] and
                pontos[0][1] < pontos[2][1] and
                pontos[0][1] < pontos[3][1] and
                pontos[0][1] < pontos[4][1] and
                pontos[0][1] < pontos[5][1] and
                pontos[0][1] < pontos[6][1] and
                pontos[0][1] < pontos[7][1] and
                pontos[0][1] < pontos[8][1] and
                pontos[0][1] < pontos[9][1] and
                pontos[0][1] < pontos[10][1] and
                pontos[0][1] < pontos[11][1] and
                pontos[0][1] < pontos[12][1] and
                pontos[0][1] < pontos[13][1] and
                pontos[0][1] < pontos[14][1] and
                pontos[0][1] < pontos[15][1] and
                pontos[0][1] < pontos[16][1] and
                pontos[0][1] < pontos[17][1] and
                pontos[0][1] < pontos[18][1] and
                pontos[0][1] < pontos[19][1] and
                pontos[0][1] < pontos[20][1] and

                pontos[8][1] > pontos[0][1] and
                pontos[8][1] > pontos[1][1] and
                pontos[8][1] > pontos[2][1] and
                pontos[8][1] > pontos[3][1] and
                pontos[8][1] > pontos[4][1] and
                pontos[8][1] > pontos[5][1] and
                pontos[8][1] > pontos[6][1] and
                pontos[8][1] > pontos[7][1] and
                pontos[8][1] > pontos[9][1] and
                pontos[8][1] > pontos[10][1] and
                pontos[8][1] > pontos[11][1] and
                pontos[8][1] > pontos[12][1] and
                pontos[8][1] > pontos[13][1] and
                pontos[8][1] > pontos[14][1] and
                pontos[8][1] > pontos[15][1] and
                pontos[8][1] > pontos[16][1] and
                pontos[8][1] > pontos[17][1] and
                pontos[8][1] > pontos[18][1] and
                pontos[8][1] > pontos[19][1] and
                pontos[8][1] > pontos[20][1] and

                # Se possível, adicionar mais condições ao ponto 4 (seja no eixo X, Y ou Z)
                pontos[4][1] > pontos[3][1]


            ):
                return "Q"


            if ( # Letra R
                pontos[8][0] < pontos[12][0] and
                pontos[6][0] > pontos[10][0] and
                pontos[20][1] > pontos[17][1] and
                pontos[16][1] > pontos[15][1]
            ):
                return "R"


            if ( # Letra S
                pontos[8][1] > pontos[5][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[20][1] > pontos[17][1] and
                pontos[4][1] < pontos[5][1] and

                pontos[4][0] < pontos[6][0] and
                pontos[4][0] < pontos[5][0]
             ):
                return "S"



          #  if ( # Letra T
                # pontos[4][2] > pontos[0][2] and
                # pontos[4][2] > pontos[1][2] and
                # pontos[4][2] > pontos[2][2] and
                # pontos[4][2] > pontos[3][2] and
                # pontos[4][2] < pontos[6][2] and
                # pontos[4][2] < pontos[7][2] and
                # pontos[4][2] < pontos[8][2] and
                # pontos[6][1] > pontos[10][1] and
                # pontos[6][1] > pontos[14][1] and
                # pontos[6][1] > pontos[19][1] and
                # pontos[7][1] > pontos[10][1] and
                # pontos[7][1] > pontos[14][1] and
                # pontos[7][1] > pontos[19][1] and
                # pontos[8][1] > pontos[10][1] and
                # pontos[8][1] > pontos[14][1] and
                # pontos[8][1] > pontos[19][1] and
                # pontos[20][1] > pontos[15][1]
            # ):
                # return "T"


            if ( # Letra U
                pontos[20][1] > pontos[17][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[12][1] < pontos[11][1] and
                pontos[8][1] < pontos[7][1] and
                abs(pontos[8][0] - pontos[12][0]) < 25 and
                pontos[4][2] < pontos[8][2]
            ):
                return "U"


            if ( # Letra V
                pontos[20][1] > pontos[17][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[12][1] < pontos[11][1] and
                pontos[8][1] < pontos[7][1] and
                abs(pontos[8][0] - pontos[12][0])  >=  25 and
                pontos[4][2] < pontos[8][2]
            ):
                return "V"


            if ( # Letra W
                pontos[8][1] < pontos[7][1] and
                pontos[12][1] < pontos[11][1] and
                pontos[16][1] < pontos[15][1] and
                pontos[20][1] > pontos[19][1] and
                abs(pontos[12][0] - pontos[8][0]) >= 25 and
                abs(pontos[16][0] - pontos[12][0]) >= 25
            ):
                return "W"


            if ( # Letra X
                pontos[8][1] < pontos[6][1] and
                pontos[20][1] > pontos[17][1] and
                pontos[16][1] > pontos[13][1] and
                pontos[12][1] > pontos[9][1] and
                pontos[12][1] > pontos[5][1] and
                abs(pontos[8][1] - pontos[10][1]) >= 50 and
                abs(pontos[8][1] - pontos[5][1]) >= 50 and
                abs(pontos[8][1] - pontos[4][1]) >= 35
            ):
                return "X"


            if ( # Letra Y
                pontos[8][1] > pontos[7][1] and
                pontos[20][1] < pontos[19][1] and
                pontos[16][1] > pontos[15][1] and
                pontos[12][1] > pontos[11][1] and
                pontos[4][0] > pontos[3][0]
            ):
                return "Y"


            if ( # Letra Z
                pontos[8][2] < pontos[0][2] and
                pontos[8][2] < pontos[1][2] and
                pontos[8][2] < pontos[2][2] and
                pontos[8][2] < pontos[3][2] and
                pontos[8][2] < pontos[4][2] and
                pontos[8][2] < pontos[5][2] and
                pontos[8][2] < pontos[6][2] and
                pontos[8][2] < pontos[7][2] and
                pontos[8][2] < pontos[9][2] and
                pontos[8][2] < pontos[10][2] and
                pontos[8][2] < pontos[12][2] and
                pontos[8][2] < pontos[11][2] and
                pontos[8][2] < pontos[13][2] and
                pontos[8][2] < pontos[14][2] and
                pontos[8][2] < pontos[15][2] and
                pontos[8][2] < pontos[16][2] and
                pontos[8][2] < pontos[18][2] and
                pontos[8][2] < pontos[17][2] and
                pontos[8][2] < pontos[19][2] and
                pontos[8][2] < pontos[20][2] and

                pontos[17][1] < pontos[18][1] and
                pontos[13][1] < pontos[14][1] and
                pontos[9][1] < pontos[10][1]
            ):
                return "Z"

            if ( # Libra Eu te amo (❤️)
                pontos[8][1] < pontos[0][1] and
                pontos[8][1] < pontos[1][1] and
                pontos[8][1] < pontos[2][1] and
                pontos[8][1] < pontos[3][1] and
                pontos[8][1] < pontos[4][1] and
                pontos[8][1] < pontos[5][1] and
                pontos[8][1] < pontos[6][1] and
                pontos[8][1] < pontos[7][1] and
                pontos[8][1] < pontos[9][1] and
                pontos[8][1] < pontos[10][1] and
                pontos[8][1] < pontos[11][1] and
                pontos[8][1] < pontos[12][1] and
                pontos[8][1] < pontos[13][1] and
                pontos[8][1] < pontos[14][1] and
                pontos[8][1] < pontos[15][1] and
                pontos[8][1] < pontos[16][1] and
                pontos[8][1] < pontos[17][1] and

                pontos[7][1] < pontos[0][1] and
                pontos[7][1] < pontos[1][1] and
                pontos[7][1] < pontos[2][1] and
                pontos[7][1] < pontos[3][1] and
                pontos[7][1] < pontos[4][1] and
                pontos[7][1] < pontos[5][1] and
                pontos[7][1] < pontos[6][1] and
                pontos[7][1] > pontos[8][1] and
                pontos[7][1] < pontos[9][1] and
                pontos[7][1] < pontos[10][1] and
                pontos[7][1] < pontos[11][1] and
                pontos[7][1] < pontos[12][1] and
                pontos[7][1] < pontos[13][1] and
                pontos[7][1] < pontos[14][1] and
                pontos[7][1] < pontos[15][1] and
                pontos[7][1] < pontos[16][1] and
                pontos[7][1] < pontos[17][1] and

                pontos[6][1] < pontos[0][1] and
                pontos[6][1] < pontos[1][1] and
                pontos[6][1] < pontos[2][1] and
                pontos[6][1] < pontos[3][1] and
                pontos[6][1] < pontos[4][1] and
                pontos[6][1] < pontos[5][1] and
                pontos[6][1] > pontos[7][1] and
                pontos[6][1] > pontos[8][1] and
                pontos[6][1] < pontos[9][1] and
                pontos[6][1] < pontos[10][1] and
                pontos[6][1] < pontos[11][1] and
                pontos[6][1] < pontos[12][1] and
                pontos[6][1] < pontos[13][1] and
                pontos[6][1] < pontos[14][1] and
                pontos[6][1] < pontos[15][1] and
                pontos[6][1] < pontos[16][1] and
                pontos[6][1] < pontos[17][1] and

                pontos[4][0] > pontos[0][0] and
                pontos[4][0] > pontos[1][0] and
                pontos[4][0] > pontos[2][0] and
                pontos[4][0] > pontos[3][0] and
                pontos[4][0] > pontos[5][0] and
                pontos[4][0] > pontos[7][0] and
                pontos[4][0] > pontos[8][0] and
                pontos[4][0] > pontos[9][0] and
                pontos[4][0] > pontos[10][0] and
                pontos[4][0] > pontos[11][0] and
                pontos[4][0] > pontos[12][0] and
                pontos[4][0] > pontos[13][0] and
                pontos[4][0] > pontos[14][0] and
                pontos[4][0] > pontos[15][0] and
                pontos[4][0] > pontos[16][0] and
                pontos[4][0] > pontos[17][0] and
                pontos[4][0] > pontos[18][0] and
                pontos[4][0] > pontos[19][0] and
                pontos[4][0] > pontos[20][0] and

                pontos[3][0] > pontos[0][0] and
                pontos[3][0] > pontos[1][0] and
                pontos[3][0] > pontos[2][0] and
                pontos[3][0] > pontos[5][0] and
                pontos[3][0] > pontos[9][0] and
                pontos[3][0] > pontos[10][0] and
                pontos[3][0] > pontos[11][0] and
                pontos[3][0] > pontos[12][0] and
                pontos[3][0] > pontos[13][0] and
                pontos[3][0] > pontos[14][0] and
                pontos[3][0] > pontos[15][0] and
                pontos[3][0] > pontos[16][0] and
                pontos[3][0] > pontos[17][0] and
                pontos[3][0] > pontos[18][0] and
                pontos[3][0] > pontos[19][0] and
                pontos[3][0] > pontos[20][0] and

                pontos[20][1] < pontos[19][1] and
                pontos[19][1] < pontos[18][1]

            ):
                return "Eu te amo"



            # Pontos das mãos (para a detecção de movimento)
            self.point_0 = pontos[0]
            self.point_1 = pontos[1]
            self.point_2 = pontos[2]
            self.point_3 = pontos[3]
            self.point_4 = pontos[4]
            self.point_5 = pontos[5]
            self.point_6 = pontos[6]
            self.point_7 = pontos[7]
            self.point_8 = pontos[8]
            self.point_9 = pontos[9]
            self.point_10 = pontos[10]
            self.point_11 = pontos[11]
            self.point_12 = pontos[12]
            self.point_13 = pontos[13]
            self.point_14 = pontos[14]
            self.point_15 = pontos[15]
            self.point_16 = pontos[16]
            self.point_17 = pontos[17]
            self.point_18 = pontos[18]
            self.point_19 = pontos[19]
            self.point_20 = pontos[20]



# Inicialização
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LeitorDeLibras()
    window.show()
    sys.exit(app.exec_())
