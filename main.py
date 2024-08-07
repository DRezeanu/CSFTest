from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QPushButton, QHBoxLayout, QComboBox, QGridLayout, 
                             QSpinBox)

from PyQt6.QtGui import QFont, QSurfaceFormat
from PyQt6.QtCore import Qt, pyqtSlot
import sys
import csv
from os.path import isfile
import os
import numpy as np
from classes import ResultsPlot, GL_CSFDemoWindow, GL_CSFTestWindow
from corefunctions import csfBestFit


class IntroWindow(QWidget):
     
    def __init__(self, parent = None):
        super(IntroWindow, self).__init__(parent)
        self.initializeWindow()
        
    def initializeWindow(self):
        self.setMinimumSize(1200,750)
        self.setWindowTitle("Welcome to NeitzCSF")
        self.setUpMainWindow()

        self.show()

    def setUpMainWindow(self):

        self.resultPlot = ResultsPlot()

        subjectInfoLabel = QLabel("Subject Info")
        subjectInfoLabel.setFont(QFont("Arial", 30))
        subjectInfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter) 

        resultsLabel = QLabel("Results")
        resultsLabel.setFont(QFont("Arial", 30))
        resultsLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nameLabel = QLabel("Name: ")
        nameLabel.setFont(QFont("Arial", 18))
        nameLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.nameText = QLineEdit()
        self.nameText.setFont(QFont("Arial", 18))
        self.nameText.setFixedHeight(30)
        self.nameText.setPlaceholderText("First Last")

        ageLabel = QLabel("Age: ")
        ageLabel.setFont(QFont("Arial", 18))
        ageLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.ageSpinBox = QSpinBox()
        self.ageSpinBox.setRange(15, 65)
        self.ageSpinBox.setFont(QFont("Arial", 18))
        self.ageSpinBox.setFixedHeight(30)
        self.ageSpinBox.setValue(25)

        ethnicityLabel = QLabel("Ethnicity: ")
        ethnicityLabel.setFont(QFont("Arial", 18))
        ethnicityLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.ethnicitySelect = QComboBox()
        ethnicity_options = ["Asian", "Black", "Hispanic", "Native American", "White", "Other"]
        self.ethnicitySelect.addItems(ethnicity_options)
        self.ethnicitySelect.setCurrentIndex(0)
        self.ethnicitySelect.setFont(QFont("Arial", 18))
        self.ethnicitySelect.setFixedHeight(30)

        stimInfoLabel = QLabel("Stimulus Parameters")
        stimInfoLabel.setFont(QFont("Arial", 30))
        stimInfoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sizeLabel = QLabel("Size (deg): ")
        sizeLabel.setFont(QFont("Arial", 18))
        sizeLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.sizeSelect = QComboBox()
        size_options = ["1", "2", "3", "4", "5"]
        self.sizeSelect.addItems(size_options)
        self.sizeSelect.setCurrentIndex(1)
        self.sizeSelect.setFont(QFont("Arial", 18))
        self.sizeSelect.setFixedHeight(30)

        eccentricityLabel = QLabel("Eccentricity (deg): ")
        eccentricityLabel.setFont(QFont("Arial", 18))
        eccentricityLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.eccentricitySelect = QComboBox()
        ecc_options = ["2", "3", "4", "5"]
        self.eccentricitySelect.addItems(ecc_options)
        self.eccentricitySelect.setCurrentIndex(0)
        self.eccentricitySelect.setFont(QFont("Arial", 18))
        self.eccentricitySelect.setFixedHeight(30)

        durationLabel = QLabel("Duration (ms): ")
        durationLabel.setFont(QFont("Arial", 18))
        durationLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.durationSelect = QComboBox()
        options = ["100", "150", "200", "250", "300", "350", "400", "450", "500"]
        self.durationSelect.addItems(options)
        self.durationSelect.setCurrentIndex(3)
        self.durationSelect.setFont(QFont("Arial", 18))
        self.durationSelect.setFixedHeight(30)

        distanceLabel = QLabel("Subject Distance (cm): ")
        distanceLabel.setFont(QFont("Arial", 18))
        distanceLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.distanceSpinBox = QSpinBox()
        self.distanceSpinBox.setRange(50, 500)
        self.distanceSpinBox.setFont(QFont("Arial", 18))
        self.distanceSpinBox.setFixedHeight(30)
        self.distanceSpinBox.setValue(250)

        startButton = QPushButton("Start")
        startButton.setFont(QFont("Arial", 18))
        startButton.clicked.connect(self.startButtonClicked)
        startButton.setMinimumSize(80, 50)
        startButton.setStyleSheet("""QPushButton {
                                     background-color: rgba(25, 25, 150, 255); color: rgba(255, 255, 255, 255); border-radius: 15px;
                                     }
                                     QPushButton::hover {
                                     background-color: rgba(25, 25, 100, 255);
                                     }
                                     QPushButton::pressed {
                                     background-color: rgba(0, 0, 0, 255);
                                     }""")

        demoButton = QPushButton("Demo")
        demoButton.setFont(QFont("Arial", 18))
        demoButton.clicked.connect(self.demoButtonClicked)
        demoButton.setMinimumSize(80, 50)
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
        
        # Add Stubject Info Inputs
        leftGrid.addWidget(subjectInfoLabel, 0, 0, 1, 3, Qt.AlignmentFlag.AlignTop)
        leftGrid.addWidget(nameLabel, 1, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.nameText, 1, 1, 2, 2)
        leftGrid.addWidget(ageLabel, 3, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.ageSpinBox, 3, 1, 2, 2)
        leftGrid.addWidget(ethnicityLabel, 5,0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.ethnicitySelect, 5, 1, 2, 2)

        # Add Stim Info Inputs
        leftGrid.addWidget(stimInfoLabel, 7, 0, 1, 3, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(durationLabel, 9, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.durationSelect, 9, 1, 2, 2)
        leftGrid.addWidget(sizeLabel, 11, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.sizeSelect, 11, 1, 2, 2)
        leftGrid.addWidget(eccentricityLabel, 13, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.eccentricitySelect, 13, 1, 2, 2)
        leftGrid.addWidget(distanceLabel, 15, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        leftGrid.addWidget(self.distanceSpinBox, 15, 1, 2, 2)
        leftGrid.addWidget(startButton, 17, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)
        
        rightGrid = QGridLayout()
        rightGrid.addWidget(resultsLabel, 0, 0, 1, 3, Qt.AlignmentFlag.AlignTop)
        rightGrid.addWidget(self.resultPlot, 1, 0, 15, 3)
        rightGrid.addWidget(demoButton, 17, 1, 1, 1, Qt.AlignmentFlag.AlignBottom)

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
                        f"{self.ethnicitySelect.currentText()}",
                        f"{self.durationSelect.currentText()}",
                        f"{self.sizeSelect.currentText()}",
                        f"{self.eccentricitySelect.currentText()}",
                        f"{self.distanceSpinBox.value()}"]

                csv_writer.writerow(data)
        else:
            with open(f"Results/{self.nameText.text()}/TestInfo.csv", 'a', newline ='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Name", "Age", "Ethnicity", "Stim Duration", "Stim Size", "Stim Eccentricity", "Subject Distance"])

                data = [f"{self.nameText.text()}",
                        f"{self.ageSpinBox.value()}",
                        f"{self.ethnicitySelect.currentText()}",
                        f"{self.durationSelect.currentText()}",
                        f"{self.sizeSelect.currentText()}",
                        f"{self.eccentricitySelect.currentText()}",
                        f"{self.distanceSpinBox.value()}"]

                csv_writer.writerow(data)

        self.testWindow = GL_CSFTestWindow(subject_distance = self.distanceSpinBox.value()*10,
                                           stim_duration = int(self.durationSelect.currentText()),
                                           stim_size = int(self.sizeSelect.currentText()),
                                           eccentricity= int(self.eccentricitySelect.currentText()))
        
        self.testWindow.finished.connect(self.plotResults)
        self.testWindow.show()

    @pyqtSlot(dict)
    def plotResults(self, results):
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

        if isfile(f"Results/{self.nameText.text()}/TestResults.csv"):
            with open(f'Results/{self.nameText.text()}/TestResults.csv', 'a', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(values)
                
        else:
            with open(f'Results/{self.nameText.text()}/TestResults.csv', 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(sfs)
                csv_writer.writerow(values)

        xvals = np.geomspace(0.4, 32, 50)
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

    def demoButtonClicked(self):
        self.demoWindow = GL_CSFDemoWindow(subject_distance=self.distanceSpinBox.value()*10)
        self.demoWindow.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


def main() -> None:
    
    # Set the OpenGL version, profile, and bits-per-channel
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

    # Create the window
    window = IntroWindow()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()