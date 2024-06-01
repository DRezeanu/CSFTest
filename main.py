from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QPushButton, QHBoxLayout, QComboBox, QGridLayout, 
                             QSpinBox, QCheckBox)

from PyQt6.QtGui import QFont, QSurfaceFormat
from PyQt6.QtCore import Qt, pyqtSlot
import sys
import csv
from os.path import isfile
import os
import numpy as np
from classes import ResultsPlot, GL_CSFDemoWindow, GL_CSFTestWindow
from functions import csfBestFit


class IntroWindow(QWidget):

    """Main window, a subclass of QWidget that allows the user to input relevant subject and test information
    and displays the results when the test ends"""
    
    def __init__(self, parent = None):
        super(IntroWindow, self).__init__(parent)
        self.initializeWindow()
        
    def initializeWindow(self):
        self.setMinimumSize(1000,625)
        self.setWindowTitle("CSF Test")
        self.setUpMainWindow()

        self.show()

    def setUpMainWindow(self):

        self.resultPlot = ResultsPlot()

        subjectInfoLabel = QLabel("Subject and Stim Info")
        subjectInfoLabel.setFont(QFont("Arial", 30))
        subjectInfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        resultsLabel = QLabel("Results")
        resultsLabel.setFont(QFont("Arial", 30))
        resultsLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nameLabel = QLabel("Name: ")
        nameLabel.setFont(QFont("Arial", 15))
        nameLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.nameText = QLineEdit()
        self.nameText.setFont(QFont("Arial", 15))
        self.nameText.setFixedHeight(25)
        self.nameText.setPlaceholderText("First Last")

        ageLabel = QLabel("Age: ")
        ageLabel.setFont(QFont("Arial", 15))
        ageLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.ageSpinBox = QSpinBox()
        self.ageSpinBox.setRange(15, 65)
        self.ageSpinBox.setFont(QFont("Arial", 15))
        self.ageSpinBox.setFixedHeight(25)
        self.ageSpinBox.setValue(25)

        startButton = QPushButton("Start")
        startButton.clicked.connect(self.startButtonClicked)
        startButton.setMinimumSize(80, 40)
        startButton.setStyleSheet("""QPushButton {
                                     background-color: rgba(25, 25, 150, 255); color: rgba(255, 255, 255, 255); border-radius: 15px;
                                     }
                                     QPushButton::hover {
                                     background-color: rgba(25, 25, 100, 255);
                                     }
                                     QPushButton::pressed {
                                     background-color: rgba(0, 0, 0, 255);
                                     }""")

        durationLabel = QLabel("Stim Duration (ms): ")
        durationLabel.setFont(QFont("Arial", 15))
        durationLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        distanceLabel = QLabel("Subject Distance (cm): ")
        distanceLabel.setFont(QFont("Arial", 15))
        distanceLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.distanceSpinBox = QSpinBox()
        self.distanceSpinBox.setRange(50, 500)
        self.distanceSpinBox.setFont(QFont("Arial", 15))
        self.distanceSpinBox.setFixedHeight(25)
        self.distanceSpinBox.setValue(100)

        self.durationSelect = QComboBox()
        options = ["100", "150", "200", "250", "300", "350", "400", "450", "500"]
        self.durationSelect.addItems(options)
        self.durationSelect.setCurrentIndex(3)
        self.durationSelect.setFont(QFont("Arial", 15))
        self.durationSelect.setFixedHeight(25)

        self.colorBlindCheckBox = QCheckBox("Are you color blind?")
        self.colorBlindCheckBox.setFont(QFont("Arial", 15))

        demoButton = QPushButton("Demo")
        demoButton.clicked.connect(self.demoButtonClicked)
        demoButton.setMinimumSize(80, 40)
        demoButton.setStyleSheet("""QPushButton {
                                     background-color: rgba(25, 25, 150, 255); color: rgba(255, 255, 255, 255); border-radius: 15px;
                                     }
                                     QPushButton::hover {
                                     background-color: rgba(25, 25, 100, 255);
                                     }
                                     QPushButton::pressed {
                                     background-color: rgba(0, 0, 0, 255);
                                     }""")

        leftGrid = QGridLayout()
        leftGrid.addWidget(subjectInfoLabel, 0, 0, 1, 3, Qt.AlignmentFlag.AlignTop)
        leftGrid.addWidget(nameLabel, 1, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.nameText, 1, 1, 2, 2)
        leftGrid.addWidget(ageLabel, 3, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.ageSpinBox, 3, 1, 2, 2)
        leftGrid.addWidget(durationLabel, 5, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.durationSelect, 5, 1, 2, 2)
        leftGrid.addWidget(distanceLabel, 7, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.distanceSpinBox, 7, 1, 2, 2)
        leftGrid.addWidget(self.colorBlindCheckBox, 9, 0, 2, 3, Qt.AlignmentFlag.AlignCenter)
        leftGrid.addWidget(startButton, 11, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)
        
        rightGrid = QGridLayout()
        rightGrid.addWidget(resultsLabel, 0, 0, 1, 3, Qt.AlignmentFlag.AlignTop)
        rightGrid.addWidget(self.resultPlot, 1, 0, 9, 3)
        rightGrid.addWidget(demoButton, 11, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)

        mainHLayout = QHBoxLayout()
        mainHLayout.addLayout(leftGrid)
        mainHLayout.addLayout(rightGrid)
        
        self.setLayout(mainHLayout)

    def startButtonClicked(self):

        if not os.path.exists("Results"):
            os.makedirs("Results")

        if not os.path.exists(f"Results/{self.nameText.text()}"):
            os.makedirs(f"Results/{self.nameText.text()}")

        if isfile(f"Results/{self.nameText.text()}/TestInfo.csv"):
            with open(f"Results/{self.nameText.text()}/TestInfo.csv", 'a', newline='') as file:
                csv_writer = csv.writer(file)
                
                data = [f"{self.nameText.text()}",
                        f"{self.ageSpinBox.value()}",
                        f"{self.distanceSpinBox.value()}",
                        f"{self.durationSelect.currentText()}",
                        f"{self.colorBlindCheckBox.isChecked()}"]

                csv_writer.writerow(data)
        else:
            with open(f"Results/{self.nameText.text()}/TestInfo.csv", 'a', newline ='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Name", "Age", "Subject Distance", "Stim Duration", "CBStatus"])

                data = [f"{self.nameText.text()}",
                        f"{self.ageSpinBox.value()}",
                        f"{self.distanceSpinBox.value()}",
                        f"{self.durationSelect.currentText()}",
                        f"{self.colorBlindCheckBox.isChecked()}"]

                csv_writer.writerow(data)

        self.testWindow = GL_CSFTestWindow(subject_distance = self.distanceSpinBox.value()*10,
                                           stim_duration = int(self.durationSelect.currentText()),
                                           stim_size = 4,
                                           eccentricity= 3)
        
        self.testWindow.finished.connect(self.plotResults)
        self.testWindow.show()

    @pyqtSlot(dict)
    def plotResults(self, results):
        """Custom "slot" method that accepts the "this test is over" signal and uses the results
        dictionary to plot the user's CSF and best fit line"""

        sfs = []
        values = []
        sortedKeys = sorted(results.keys())

        for key in sortedKeys:
            sfs.append(key)
            values.append(np.mean(results[key]))

        for i in range(len(values)):
            values[i]=1/values[i]

        sfs = np.asarray(sfs)
        values = np.asarray(values)

        xvals = np.geomspace(0.2, 32, 50)
        bestFit = csfBestFit(xvals, sfs, values)

        self.resultPlot.axes.clear()
        
        self.resultPlot.axes.scatter(sfs, values, marker = 's', color='k', label = "Data")
        self.resultPlot.axes.plot(xvals, bestFit, "-r", label="Best Fit", linewidth = 2)
        self.resultPlot.axes.set_xscale('log')
        self.resultPlot.axes.set_yscale('log')
        self.resultPlot.axes.set_ylim([1, 200])
        self.resultPlot.axes.set_xlim(([0.1, 50]))
        self.resultPlot.axes.set_yticks([1, 10, 50, 100, 200], ['1', '10', '50', '100', '200'])
        self.resultPlot.axes.set_xticks([0.1, 0.5, 1, 2, 4, 8, 16, 32], ['0.1', '0.5', '1', '2', '4', '8', '16', '32'])
        self.resultPlot.axes.set_xlabel("Spatial Frequency (c/deg)")
        self.resultPlot.axes.set_ylabel("Contrast Sensitivity")
        self.resultPlot.axes.set_title(f"{self.nameText.text()} CSF Results")
        self.resultPlot.axes.legend()
        self.resultPlot.axes.grid(True)
        self.resultPlot.draw()

        directory = f"Results/{self.nameText.text()}"
        numPlots = 0
        for filename in os.listdir(directory):
            if "Plot" in filename:
                numPlots+= 1

        if numPlots:        
            self.resultPlot.print_png(f"Results/{self.nameText.text()}/Plot_{numPlots}.png")
        else:
            self.resultPlot.print_png(f"Results/{self.nameText.text()}/Plot.png")

        if isfile(f"Results/{self.nameText.text()}/TestResults.csv"):
            with open(f'Results/{self.nameText.text()}/TestResults.csv', 'a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(values)
                
        else:
            with open(f'Results/{self.nameText.text()}/TestResults.csv', 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(sfs)
                csv_writer.writerow(values)

    def demoButtonClicked(self):
        self.demoWindow = GL_CSFDemoWindow(subject_distance=self.distanceSpinBox.value()*10,
                                           stim_size = 4,
                                           eccentricity = 3)
        self.demoWindow.show()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


def main() -> None:
    
    # Set the OpenGL version, profile, and bits-per-color-channel
    format_10bit = QSurfaceFormat()
    format_10bit.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    format_10bit.setVersion(3,3)
    format_10bit.setBlueBufferSize(10)
    format_10bit.setGreenBufferSize(10)
    format_10bit.setRedBufferSize(10)
    format_10bit.setAlphaBufferSize(2)
    QSurfaceFormat.setDefaultFormat(format_10bit)

    # Initialize the application
    app = QApplication(sys.argv)

    # Create the main window
    window = IntroWindow()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()