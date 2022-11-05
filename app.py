import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel, QHBoxLayout


# Subclass QMainWindow to customize your application's main window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")


        only_angles = QIntValidator(-(360*4), 360*4)
        xangle = QLineEdit("Insira o angulo do eixo X")
        yangle = QLineEdit("Insira o angulo do eixo Y")
        zangle = QLineEdit("Insira o angulo do eixo Z")

        xangle.setValidator(only_angles)
        yangle.setValidator(only_angles)

        xlabel = QLabel("X angle")
        ylabel = QLabel("Y angle")
        zlabel = QLabel("Z angle")

        hLayout1 = QHBoxLayout()
        hLayout1.addWidget(xlabel)
        hLayout1.addWidget(xangle)
        hLayout2 = QHBoxLayout()
        hLayout2.addWidget(ylabel)
        hLayout2.addWidget(yangle)
        hLayout3 = QHBoxLayout()
        hLayout3.addWidget(zlabel)
        hLayout3.addWidget(zangle)

        # button.setCheckable(True)
        # button.clicked.connect(self.the_button_was_clicked)
        # button.clicked.connect(self.the_button_was_toggled)

        layout = QVBoxLayout()
        layout.addLayout(hLayout1)
        layout.addLayout(hLayout2)
        layout.addLayout(hLayout3)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # def the_button_was_clicked(self, some):
    #     print("Clicked!", some)
    #
    # def the_button_was_toggled(self, checked):
    #     print("Checked?", checked)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
