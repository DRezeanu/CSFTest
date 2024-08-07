from corefunctions import (deg2pix, getScreenDims, createShaderProgram,
                           genQuadWithTextureCoords, genVAOandVBOWithTextureCoords,
                           genTextureFromImage, csfParabola, csfBestFit, pix2deg,
                           getNyquist)
import ctypes
from OpenGL import GL
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtOpenGL import QOpenGLWindow
import numpy as np
import numpy.random as random
import time

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

## OpenGL Windows and Stimulus Classes ##

class GLImageStim:

    def __init__(self, filename, width, height, x_offset = 0, y_offset = 0, subject_distance = 1000):
        
        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        self.filename = filename
        self.width = width
        self.height = height
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])

        self.quad_width = deg2pix(self.width, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.quad_height = deg2pix(self.height, self.subject_distance, physical_dims[0], pixel_dims[0])

        self.vertices, self.vertex_count = genQuadWithTextureCoords(self.quad_width, self.quad_height, pixel_dims[0], 
                                                              pixel_dims[1], self.x_offset, self.y_offset)
        
        self.texture = genTextureFromImage(self.filename)
        
        # Create vao and vbo
        self.vao, self.vbo = genVAOandVBOWithTextureCoords(self.vertices)

        # Create shader program
        self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/image_frag_shader.txt")
        self.tex_uniform = GL.glGetUniformLocation(self.shader_program, "imageTexture")

        GL.glUseProgram(self.shader_program)
        GL.glUniform1i(self.tex_uniform, 0)

    def use(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glBindVertexArray(self.vao)
        GL.glUseProgram(self.shader_program)
        GL.glUniform1i(self.tex_uniform, 0)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)


class GLMaskStim:
    def __init__(self, size = 3, x_offset = 0, y_offset = 0, subject_distance = 1000):

        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        self.size = size
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.quad_size = deg2pix(self.size, self.subject_distance, physical_dims[0], pixel_dims[0])

        self.vertices, self.vertex_count = genQuadWithTextureCoords(quad_width = self.quad_size,
                                                               quad_height = self.quad_size,
                                                               screen_width = pixel_dims[0],
                                                               screen_height = pixel_dims[1],
                                                               x_offset = self.x_offset,
                                                               y_offset = self.y_offset)
        
        # Create vao and vbo
        self.vao, self.vbo = genVAOandVBOWithTextureCoords(self.vertices)

        # Create shader program
        self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/mask_frag_shader.txt")

        GL.glUseProgram(self.shader_program)


    def use(self):
        GL.glUseProgram(self.shader_program)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)

class GLCircleStim:
    def __init__(self, size = 4, x_offset = 0, y_offset = 0, subject_distance = 1000):

        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        self.size = size
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.quad_size = deg2pix(self.size, self.subject_distance, physical_dims[0], pixel_dims[0])

        self.vertices, self.vertex_count = genQuadWithTextureCoords(quad_width = self.quad_size,
                                                               quad_height = self.quad_size,
                                                               screen_width = pixel_dims[0],
                                                               screen_height = pixel_dims[1],
                                                               x_offset = self.x_offset,
                                                               y_offset = self.y_offset)
        
        # Create vao and vbo
        self.vao, self.vbo = genVAOandVBOWithTextureCoords(self.vertices)

        # Create shader program
        self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/circle_frag_shader.txt")

        GL.glUseProgram(self.shader_program)


    def use(self):
        GL.glUseProgram(self.shader_program)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)


class GLGratingStim:

    def __init__(self, size = 4, x_offset = 0, y_offset = 0, subject_distance = 1000,
                 sf = 4, ori = 0, contrast = 0.1, phase = 0, sd = 0.15, wave = 'sin'):
        
        """OpenGL gabor stimulus class with given size, location, spatial frequency, orientation
        contrast, and phase."""

        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        self.wave = wave
        self.size = size
        self.sd = sd
        self.sf = sf
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.ori = ori
        self.contrast = contrast
        self.phase = phase

        self.quad_size = deg2pix(self.size, self.subject_distance, physical_dims[0], pixel_dims[0])

        self.vertices, self.vertex_count = genQuadWithTextureCoords(quad_width = self.quad_size,
                                                               quad_height = self.quad_size,
                                                               screen_width = pixel_dims[0],
                                                               screen_height = pixel_dims[1],
                                                               x_offset = self.x_offset,
                                                               y_offset = self.y_offset)
        
        # Create vao and vbo
        self.vao, self.vbo = genVAOandVBOWithTextureCoords(self.vertices)

        # Create shader program and requisite uniform variables for stim creation
        if self.wave == 'sqr':
            self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/square_wave_frag_shader.txt")
        else:
            self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/gabor_frag_shader.txt")

        self.u_orientation = GL.glGetUniformLocation(self.shader_program, "u_orientation")
        self.u_contrast = GL.glGetUniformLocation(self.shader_program, "u_contrast")
        self.u_sf = GL.glGetUniformLocation(self.shader_program, "u_spatial_frequency")
        self.u_phase = GL.glGetUniformLocation(self.shader_program, "u_phase")

        self.sf *= self.size

        GL.glUseProgram(self.shader_program)
        GL.glUniform1f(self.u_orientation, self.ori)
        GL.glUniform1f(self.u_contrast, self.contrast)
        GL.glUniform1f(self.u_sf, self.sf)
        GL.glUniform1f(self.u_phase, self.phase)

    def use(self):
        GL.glUseProgram(self.shader_program)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)


class GL_CSFDemoWindow(QOpenGLWindow):

    def __init__(self, subject_distance, stim_size = 2, eccentricity = 2, parent = None):
        super(GL_CSFDemoWindow, self).__init__(parent)
        self.subject_distance = subject_distance
        self.stim_size = stim_size
        self.eccentricity = eccentricity

        self.confirmation_beep = QSoundEffect()
        self.confirmation_beep.setSource(QUrl.fromLocalFile("Assets/confirmation.wav"))

    def initializeGL(self) -> None:

        self.page1 = True
        self.page_correct = False
        self.page_wrong = False
        self.show_fixation = False

        self.YN_active = False
        self.space_active = True
        self.arrows_active = False

        GL.glClearColor(0.5**(1/2.42), 0.5**(1/2.42), 0.5**(1/2.42), 1.0)

        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        degree_dims = [pix2deg(pixel_dims[0], self.subject_distance, physical_dims[0], pixel_dims[0]), 
                       pix2deg(pixel_dims[1], self.subject_distance, physical_dims[0], pixel_dims[0])]

        if self.context().format().redBufferSize() == 8:
            QMessageBox.critical(None, "Color Depth Warning", """Currently running in 8-bit color mode.
                                Maximum contrast limited!""", QMessageBox.StandardButton.Ok)

        # Create controller classes
        self.demoController = DemoController(num_stims=4)
        
        # Generate gabors and fixation
        self.top_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = self.eccentricity)
        self.bottom_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = -self.eccentricity)
        self.right_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = self.eccentricity, y_offset = 0)
        self.left_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = -self.eccentricity, y_offset = 0)

        self.fixation = GLImageStim("Assets/fixation_hash.png", width = 0.5, height = 0.5, subject_distance = self.subject_distance)
        self.explanation = GLImageStim("Assets/explanation_window.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.press_space = GLImageStim("Assets/begin_demo.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.correct_message = GLImageStim("Assets/correct.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.wrong_message = GLImageStim("Assets/wrong.png", width = degree_dims[1], height = degree_dims[1], subject_distance=self.subject_distance)

        # Sync screen repaint to vertical refresh rate
        self.frameSwapped.connect(self.update)

        self.correct = 0

        # Enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Set display to full screen
        self.showFullScreen()
    
    def resizeGL(self, w: int, h: int) -> None:
        return super().resizeGL(w, h)
    
    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        if self.page1:
            self.explanation.use()

        if self.show_fixation:
            self.fixation.use()

        if self.demoController.trigger[1]:
            self.top_gabor.use()
            self.correct = 1

        if self.demoController.trigger[2]:
            self.right_gabor.use()
            self.correct = 2

        if self.demoController.trigger[3]:
            self.bottom_gabor.use()
            self.correct = 3

        if self.demoController.trigger[4]:
            self.left_gabor.use()
            self.correct = 4

        if self.page_correct:
            self.correct_message.use()

        if self.page_wrong:
            self.wrong_message.use()
          
    def keyPressEvent(self, event) -> None:
        self.makeCurrent()
        if event.key() == Qt.Key.Key_Escape:
            self.close()

        if event.key() == Qt.Key.Key_Space:
            if self.space_active:
                self.page1 = False
                self.show_fixation = True
                self.shuffleAttributes()
                self.demoController.next()
                self.space_active = False
                self.arrows_active = True

        if event.key() == Qt.Key.Key_Y:
            if self.YN_active:
                self.YN_active = False
                self.page1 = False
                self.page_correct = False
                self.page_wrong = False
                self.show_fixation = True
                self.shuffleAttributes()
                self.demoController.next()
                self.arrows_active = True

        if event.key() == Qt.Key.Key_N:
            if self.YN_active:
                self.close()

        if event.key() == Qt.Key.Key_Up:
            if self.arrows_active:
                self.confirmation_beep.play()
                self.arrows_active = False
                if self.correct == 1:
                    self.show_fixation = False
                    self.page_correct = True
                else:
                    self.show_fixation = False
                    self.page_correct = False
                    self.page_wrong = True
                self.YN_active = True
        
        if event.key() == Qt.Key.Key_Right:
            if self.arrows_active:
                self.confirmation_beep.play()
                self.arrows_active = False 
                if self.correct == 2:
                    self.show_fixation = False
                    self.page_correct = True
                else:
                    self.show_fixation = False
                    self.page_correct = False
                    self.page_wrong = True
                self.YN_active = True
                
        if event.key() == Qt.Key.Key_Down:
            if self.arrows_active:
                self.confirmation_beep.play()
                self.arrows_active = False
                if self.correct == 3:
                    self.show_fixation = False
                    self.page_correct = True
                else:
                    self.show_fixation = False
                    self.page_correct = False
                    self.page_wrong = True
                self.YN_active = True
        
        if event.key() == Qt.Key.Key_Left:
            if self.arrows_active:
                self.confirmation_beep.play()
                self.arrows_active = False
                if self.correct == 4:
                    self.show_fixation = False
                    self.page_correct = True
                else:
                    self.show_fixation = False
                    self.page_correct = False
                    self.page_wrong = True
                self.YN_active = True
        
        if event.key() == Qt.Key.Key_Z:
            if self.arrows_active:
                self.confirmation_beep.play()
                self.arrows_active = False
                self.show_fixation = False
                self.page_correct = False
                self.page_wrong = True
            self.YN_active = True

    def shuffleAttributes(self):
        # shuffle contrast
        contrast = random.choice([0.08, 0.16, 0.32, 0.64])

        GL.glUseProgram(self.top_gabor.shader_program)
        GL.glUniform1f(self.top_gabor.u_contrast, contrast)

        GL.glUseProgram(self.bottom_gabor.shader_program)
        GL.glUniform1f(self.bottom_gabor.u_contrast, contrast)

        GL.glUseProgram(self.right_gabor.shader_program)
        GL.glUniform1f(self.right_gabor.u_contrast, contrast)

        GL.glUseProgram(self.left_gabor.shader_program)
        GL.glUniform1f(self.left_gabor.u_contrast, contrast)

        # set orientation

        GL.glUseProgram(self.top_gabor.shader_program)
        GL.glUniform1f(self.top_gabor.u_orientation, 0)

        GL.glUseProgram(self.bottom_gabor.shader_program)
        GL.glUniform1f(self.bottom_gabor.u_orientation, 0)

        GL.glUseProgram(self.right_gabor.shader_program)
        GL.glUniform1f(self.right_gabor.u_orientation, 90)

        GL.glUseProgram(self.left_gabor.shader_program)
        GL.glUniform1f(self.left_gabor.u_orientation, 90)

        # shuffle sf
        sf = random.choice([2, 4, 6, 8, 16])
        sf = sf*self.stim_size

        GL.glUseProgram(self.top_gabor.shader_program)
        GL.glUniform1f(self.top_gabor.u_sf, sf)

        GL.glUseProgram(self.bottom_gabor.shader_program)
        GL.glUniform1f(self.bottom_gabor.u_sf, sf)

        GL.glUseProgram(self.left_gabor.shader_program)
        GL.glUniform1f(self.left_gabor.u_sf, sf)

        GL.glUseProgram(self.right_gabor.shader_program)
        GL.glUniform1f(self.right_gabor.u_sf, sf)

    def close(self):
        self.top_gabor.destroy()
        self.left_gabor.destroy()
        self.right_gabor.destroy()
        self.bottom_gabor.destroy()
        self.fixation.destroy()
        self.explanation.destroy()
        self.press_space.destroy()
        self.wrong_message.destroy()
        self.correct_message.destroy()
        super().close()


class GL_CSFTestWindow(QOpenGLWindow):

    finished=pyqtSignal(dict)

    def __init__(self, subject_distance, stim_duration = 250, stim_size = 2, eccentricity = 2, parent=None):
        super(GL_CSFTestWindow, self).__init__(parent)
        self.subject_distance = subject_distance
        self.eccentricity = eccentricity
        self.stim_size = stim_size
        self.duration = stim_duration
        self.confirmation_beep = QSoundEffect()
        self.confirmation_beep.setSource(QUrl.fromLocalFile("Assets/confirmation.wav"))

        self.keyPressTimer = QTimer()
        self.keyPressTimer.setTimerType(Qt.TimerType.PreciseTimer)
        self.keyPressTimer.setSingleShot(True)
        self.keyPressTimer.setInterval(1750)
        self.keyPressTimer.timeout.connect(self.activateArrows)

    def initializeGL(self) -> None:

        self.spaceActive = True
        self.arrowsActive = False

        self.first_page = True
        self.show_stim = False
        self.show_fixation = False

        GL.glClearColor(0.5**(1/2.42), 0.5**(1/2.42), 0.5**(1/2.42), 1.0)

        pixel_ratio, pixel_dims, physical_dims = getScreenDims()
        nyquist = getNyquist(physical_dims[0], pixel_dims[0], self.subject_distance)
        print(nyquist)

        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        degree_dims = [pix2deg(pixel_dims[0], self.subject_distance, physical_dims[0], pixel_dims[0]), 
                       pix2deg(pixel_dims[1], self.subject_distance, physical_dims[0], pixel_dims[0])]

        if self.context().format().redBufferSize() == 8:
            QMessageBox.critical(None, "Color Depth Warning", """Currently running in 8-bit color mode.
                                Maximum contrast limited!""", QMessageBox.StandardButton.Ok)
        
        # Generate gabors and fixation
        self.top_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = self.eccentricity)
        self.bottom_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = -self.eccentricity)
        self.right_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = self.eccentricity, y_offset = 0)
        self.left_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = -self.eccentricity, y_offset = 0)

        self.fixation = GLImageStim("Assets/fixation_hash.png", width = 0.5, height = 0.5, subject_distance = self.subject_distance)
        self.explanation = GLImageStim("Assets/start_window.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.break_time = GLImageStim("Assets/break_time.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.test_over = GLImageStim("Assets/all_done.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)

        self.top_mask = GLMaskStim(size = self.stim_size, subject_distance = self.subject_distance, x_offset = 0, y_offset = self.eccentricity)
        self.bottom_mask = GLMaskStim(size = self.stim_size, subject_distance = self.subject_distance, x_offset = 0, y_offset = -self.eccentricity)
        self.right_mask = GLMaskStim(size = self.stim_size, subject_distance = self.subject_distance, x_offset = self.eccentricity, y_offset = 0)
        self.left_mask = GLMaskStim(size = self.stim_size, subject_distance=self.subject_distance, x_offset = -self.eccentricity, y_offset = 0)

        # Create controller classes
        self.trialHandler = TrialHandler(stim_size = self.stim_size)
        
        if self.trialHandler.sfMax > nyquist:
            raise ValueError("Max spatial frequency exceeds nyquist limit for this display and disatnce")
        
        self.displayHandler = DisplayHandler(num_stims = 4, stim_duration = self.duration, pre_stim_interval=1500)

        # Add all shaders and uniforms to list

        self.gabor_shaders = [self.top_gabor.shader_program,
                                self.right_gabor.shader_program,
                                self.bottom_gabor.shader_program,
                                self.left_gabor.shader_program]
        
        self.gabor_uniforms = [[self.top_gabor.u_sf, self.top_gabor.u_orientation, self.top_gabor.u_phase, self.top_gabor.u_contrast],
                               [self.right_gabor.u_sf, self.right_gabor.u_orientation, self.right_gabor.u_phase, self.right_gabor.u_contrast],
                               [self.bottom_gabor.u_sf, self.bottom_gabor.u_orientation, self.right_gabor.u_phase, self.bottom_gabor.u_contrast],
                               [self.left_gabor.u_sf, self.left_gabor.u_orientation, self.left_gabor.u_phase, self.bottom_gabor.u_contrast]]

        # Sync screen repaint to vertical refresh rate
        self.frameSwapped.connect(self.update)

        self.userInput = None

        # Enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Set display to full screen
        self.showFullScreen()

    def resizeGL(self, w: int, h: int) -> None:
        return super().resizeGL(w, h)
    
    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self.trialHandler.trialOver:
            self.break_time.use()
            self.spaceActive = True
            self.arrowsActive = False
        elif self.trialHandler.testOver:
            self.test_over.use()
            self.spaceActive = True
            self.arrowsActive = False
        elif self.first_page:
            self.explanation.use()
        else:
            if self.show_fixation:
                self.fixation.use()

            if self.displayHandler.trigger[0]:
                self.top_gabor.use()
            else:
                self.top_mask.use()
            
            if self.displayHandler.trigger[1]:
                self.right_gabor.use()
            else:
                self.right_mask.use()
            
            if self.displayHandler.trigger[2]:
                self.bottom_gabor.use()
            else:
                self.bottom_mask.use()

            if self.displayHandler.trigger[3]:
                self.left_gabor.use()
            else:
                self.left_mask.use()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        
        if event.key() == Qt.Key.Key_Space:
            if self.spaceActive:
                if self.first_page:
                    self.first_page = False
                    self.show_fixation = True
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.current_stim_params
                    if self.displayHandler.currentStim == 0 or self.displayHandler.currentStim == 2:
                        stim_params[1] = 0
                    else:
                        stim_params[1] = 90
                    self.displayHandler.showStim(stim_params, shader, uniforms)
                    self.arrowsActive = True
                    self.spaceActive = False
                elif self.trialHandler.testOver:
                    self.finished.emit(self.trialHandler.results)
                    self.close()
                else:
                    self.trialHandler.trialOver = False
                    self.show_fixation = True
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.current_stim_params
                    if self.displayHandler.currentStim == 0 or self.displayHandler.currentStim == 2:
                        stim_params[1] = 0
                    else:
                        stim_params[1] = 90
                    self.displayHandler.showStim(stim_params, shader, uniforms)
                    self.arrowsActive = True
                    self.spaceActive = False
                    time.sleep(0.5)
                    

        if event.key() == Qt.Key.Key_Up:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 0:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.keyPressTimer.start()


        if event.key() == Qt.Key.Key_Right:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 1:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.keyPressTimer.start()

        if event.key() == Qt.Key.Key_Down:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 2:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.keyPressTimer.start()


        if event.key() == Qt.Key.Key_Left:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 3:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0, self.displayHandler.currentStim)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.keyPressTimer.start()
        
        if event.key() == Qt.Key.Key_Z:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                stim_params = self.trialHandler.nextStim(0, self.displayHandler.currentStim)
                if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                    self.displayHandler.showStim(stim_params, shader, uniforms)
                self.keyPressTimer.start()

    def activateArrows(self):
        self.arrowsActive = True

    def close(self):
        self.top_gabor.destroy()
        self.left_gabor.destroy()
        self.right_gabor.destroy()
        self.bottom_gabor.destroy()
        self.fixation.destroy()
        self.explanation.destroy()
        self.break_time.destroy()
        self.top_mask.destroy()
        self.bottom_mask.destroy()
        self.left_mask.destroy()
        self.right_mask.destroy()
        super().close()

## Stimulus and Trial Controllers ##

class DemoController:
    
    def __init__(self, num_stims = 4, stim_duration = 250, countdown_time = 2000):
        
        self.num_stims = num_stims
        self.stim_duration = stim_duration
        self.countdown_time = countdown_time
        self.current_trigger = 1
        self.trigger = {}

        self.beep = QSoundEffect()
        self.beep.setSource(QUrl.fromLocalFile("Assets/beep.wav"))

        for i in range(1,num_stims+1):
            self.trigger[i] = False

        self.countdown_timer = QTimer()
        self.countdown_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.countdown_timer.setSingleShot(True)
        self.countdown_timer.setInterval(self.countdown_time)
        self.countdown_timer.timeout.connect(self.showStim)
        

        self.stim_timer = QTimer()
        self.stim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.stim_timer.setSingleShot(True)
        self.stim_timer.setInterval(self.stim_duration)
        self.stim_timer.timeout.connect(self.hideStim)


    def next(self):
        self.current_trigger = np.random.choice([1, 2, 3, 4])
        self.countdown_timer.start()

    def showStim(self):
        self.beep.play()
        self.trigger[self.current_trigger] = True
        self.stim_timer.start()

    def hideStim(self):
        self.trigger[self.current_trigger] = False


class singleStaircaseController:
    def __init__(self, startVal: float, nReversals: int, scale: str):

        self._startVal = startVal
        self._nReversals = nReversals
        self._numWrongInARow = 0
        
        if "log" in scale:
            self._stepSizes = np.logspace(0.3, 0.0075, nReversals)
        else:
            self._stepSizes = np.linspace(0.5, 0.0015, nReversals)

        self._scale = scale
        self._countReversals = 0
        self._reversalValues = []
        self._currentStep = self._stepSizes[0]
        self._isReversal = False
        self.testOver = False
        self.result = None

        self._previousAnswer = None
        self.currentValue = self._startVal

    def next(self, userInput: bool):

        if not userInput:
            self._numWrongInARow += 1
        else:
            self._numWrongInARow = 0

        if self._numWrongInARow > 9 and self.currentValue > 0.8:
            self.result = 1.0
            self.testOver = True
            return self.result

        self._isReversal = self.checkReversal(userInput)

        if self._countReversals >= self._nReversals+1:
            self._reversalValues.pop(0)
            self._reversalValues.pop(0)
            self.result = np.mean(self._reversalValues)
            self.testOver = True
            return self.result
            
                
        if self._isReversal:
            self._reversalValues.append(self.currentValue)
            self.decreaseStepSize()
            self._isReversal = False
        
        if "log" in self._scale:
            if userInput:
                self.currentValue = np.max([self.currentValue/self._currentStep, 0])
            else:
                self.currentValue = np.min([self.currentValue*self._currentStep, 1.0])
        else:
            if userInput:
                self.currentValue = np.max([self.currentValue-self._currentStep, 0])
            else:
                self.currentValue = np.min([self.currentValue+self._currentStep, 1.0])

        return self.currentValue
    
    def decreaseStepSize(self):
        
        idx = np.where(self._stepSizes==self._currentStep)[0][0]

        if idx < len(self._stepSizes)-1:
            self._currentStep = self._stepSizes[idx+1]
        else:
            self._currentStep = self._stepSizes[-1]

    def checkReversal(self, input):
        if self._previousAnswer == None:
            self._previousAnswer = input
            return False
        elif self._previousAnswer != input:
            self._previousAnswer = input
            self._countReversals += 1
            return True
        else:
            self._previousAnswer = input
            return False
        
    def reset(self, startVal):
        self._startVal = startVal
        self._countReversals = 0
        self._reversalValues = []
        self._currentStep = self._stepSizes[0]
        self.testOver = False
        self._isReversal = False
        self._previousAnswer = None
        self.currentValue = self._startVal


class multiStaircaseController:
    

    def __init__(self, numStaircases: int, startVals: list[float], nReversals: int, scale: str):
        
        self.numStaircases = numStaircases
        self.startVals = startVals
        self.nReversals = nReversals

        if len(startVals) != numStaircases:
            raise ValueError("You must have the same number of start values as staircases")

        # Initialize all staircases
        self.staircases: list[singleStaircaseController] = []
        for i in range(numStaircases):
            self.staircases.append(singleStaircaseController(startVals[i], nReversals, scale))

        # Set important variables
        self.allDone = False
        self.firstRound = True
        choices = self.identifyValidStaircases()
        self.current_staircase = None
        self.first_staircase = self.staircases[np.random.choice(choices)]
        self.currentValue = self.first_staircase.currentValue
        self.results = None

        

    def next(self, userInput):
        # Check which staircases are still active and only chose from those
        choices = self.identifyValidStaircases()

        if not choices:
            self.allDone = True
            self.results = self.getResults()
            return np.mean(self.results)
        
        if self.firstRound:
            self.first_staircase.next(userInput)
            self.current_staircase = np.random.choice(choices)
            self.firstRound = False
        else:
            self.staircases[self.current_staircase].next(userInput)
            self.current_staircase = np.random.choice(choices)
            
        self.currentValue = self.staircases[self.current_staircase].currentValue

        return self.currentValue

    
    def identifyValidStaircases(self):
        choices = []
        for i in range(len(self.staircases)):
            if not self.staircases[i].testOver:
                choices.append(i)

        return choices
            

    def getResults(self):
        results = []
        for i in range(len(self.staircases)):
            results.append(self.staircases[i].result)

        return results
    
    def reset(self, startVals: list[float]):

        if len(startVals) != self.numStaircases:
            raise ValueError("You must have the same number of start values as staircases")
        
        for i in range(len(self.staircases)):
            self.staircases[i].reset(startVals[i])
        
        # Reset important variables
        self.allDone = False
        self.firstRound = True
        choices = self.identifyValidStaircases()
        self.current_staircase = None
        self.first_staircase = self.staircases[np.random.choice(choices)]
        self.currentValue = self.first_staircase.currentValue
        self.results = None

    
class TrialHandler:

    def __init__(self, stim_size, sfMin = 0.5, sfMax = 32, numTrials: int = 13, numStaircases: int = 2, scale: str = "log",
                 nReversals: int = 7):
        
        self.sfMin = sfMin
        self.sfMax = sfMax
        self.stim_size = stim_size
        self.numTrials = numTrials
        self.numStaircases = numStaircases
        self.scale = scale
        self.nReversals = nReversals
        self.currentTrial = 0
        self.results={}
        self.phase = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        self.ori = np.random.choice([225, 225])
        self.testOver = False
        self.trialOver = False

        # Generate list of spatial frequencies to test
        self.SFs = np.geomspace(sfMin, sfMax, self.numTrials)

        # Shuffle their order in place
        np.random.shuffle(self.SFs)

        # generate the expected values given an average CSF, and set start values for staircases accordingly (ignoring this for now)
        expected_values = csfParabola(self.SFs, 200, 3, 3.5, 20)
        self.startVals = np.empty((self.numTrials, self.numStaircases))

        for i in range(len(expected_values)):
            # self.startVals[i,:] = [np.max([expected_values[i]*2,1]), np.max([expected_values[i]/2,1])]
            self.startVals[i,:] = [0.005, 0.8]

        self.staircaseHandler = self.genStaircase(self.currentTrial)

        # Set current stim parameters
        self.current_stim_params = [self.SFs[self.currentTrial]*self.stim_size, self.ori, self.phase, self.staircaseHandler.currentValue]
    
    # Stim Param 0 = SF, Stim Param 1 = Orientation, Stim Param 2 = Phase, Stim Param 3 = Contrast
    def nextStim(self, userInput, stimLocation):
        if self.staircaseHandler.results: 
            self.trialOver = True
            self.results[self.SFs[self.currentTrial]] = self.staircaseHandler.results

        if self.trialOver and self.currentTrial == self.numTrials-1:
            self.testOver = True
            self.trialOver = False

        elif self.trialOver and self.currentTrial < self.numTrials-1:
            self.currentTrial +=1
            self.staircaseHandler.reset(self.startVals[self.currentTrial,:])
            newVal = self.staircaseHandler.currentValue
            self.current_stim_params[0] = self.SFs[self.currentTrial]*self.stim_size
            if stimLocation == 0 or stimLocation == 2:
                self.current_stim_params[1] = 0
            else:
                self.current_stim_params[1] = 90
            self.current_stim_params[2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            self.current_stim_params[3] = newVal
        else:
            newVal = self.staircaseHandler.next(userInput)
            if stimLocation == 0 or stimLocation == 2:
                self.current_stim_params[1] = 0
            else:
                self.current_stim_params[1] = 90

            self.current_stim_params[2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            self.current_stim_params[3] = newVal
        
        return self.current_stim_params


    def genStaircase(self, currentTrial):

        staircase = multiStaircaseController(self.numStaircases,
                                                         self.startVals[currentTrial,:],
                                                         self.nReversals, self.scale)
        return staircase


class DisplayHandler:

    def __init__(self, num_stims, stim_duration, pre_stim_interval = 1000):

        self.numStims = num_stims
        self.showFixation = False
        self.trigger = {}
        for i in np.arange(0,num_stims):
            self.trigger[i] = False
        self.currentStim = None

        self.stim_timer = QTimer()
        self.stim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.stim_timer.setSingleShot(True)
        self.stim_timer.setInterval(stim_duration)

        self.wait_timer = QTimer()
        self.wait_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.wait_timer.setSingleShot(True)
        self.wait_timer.setInterval(pre_stim_interval)

        self.stim_timer.timeout.connect(self.showInterStim)
        self.wait_timer.timeout.connect(self.makeVisible)

        self.beep = QSoundEffect()
        self.beep.setSource(QUrl.fromLocalFile("Assets/beep.wav"))

    def pickStim(self, shaderArray, uniformArray):
        self.currentStim = np.random.choice(np.arange(0, self.numStims))
        shader = shaderArray[self.currentStim]
        uniforms = uniformArray[self.currentStim]

        return shader, uniforms


    def showStim(self, stim_parameters, shader, uniforms):
        sf = stim_parameters[0]
        ori = stim_parameters[1]
        phase = stim_parameters[2]
        contrast = stim_parameters[3]

        GL.glUseProgram(shader)
        GL.glUniform1f(uniforms[0], sf)
        GL.glUniform1f(uniforms[1], ori)
        GL.glUniform1f(uniforms[2], phase)
        GL.glUniform1f(uniforms[3], contrast)

        self.wait_timer.start()

    def makeVisible(self):
        self.beep.play()
        self.trigger[self.currentStim] = True
        self.stim_timer.start()
    
    def showInterStim(self):
        self.trigger[self.currentStim] = False


## Plotting Class ##

class ResultsPlot(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(ResultsPlot, self).__init__(fig)
        self.setPlaceholderResults()

    def setPlaceholderResults(self):
        xvals = np.geomspace(0.4, 32, 50)
        sampleSFs = np.geomspace(0.5, 32, 13)
        sampleData = np.asarray([60, 82, 108, 132, 149, 161, 152, 126, 90, 43, 23, 8, 2.3])

        bestFit = csfBestFit(xvals, sampleSFs, sampleData)

        self.axes.scatter(sampleSFs, sampleData, marker = 's', color='k', label = "Sample Data")
        self.axes.plot(xvals, bestFit, "-r", label="Best Fit", linewidth = 2)
        self.axes.set_xscale('log')
        self.axes.set_yscale('log')
        self.axes.set_ylim([1,200])
        self.axes.set_xlim(([0.1, 50]))
        self.axes.set_yticks([1, 10, 50, 100, 200], ['1', '10', '50', '100', '200'])
        self.axes.set_xticks([0.1, 0.5, 1, 2, 4, 8, 16, 32], ['0.1', '0.5', '1', '2', '4', '8', '16', '32'])
        self.axes.set_xlabel("Spatial Frequency (c/deg)")
        self.axes.set_ylabel("Contrast Sensitivity")
        self.axes.set_title("Sample CSF Results")
        self.axes.legend()
        self.axes.grid(True)
        self.draw()


def testStaircase(scaletype: str, start: float, reversals: int, true_val: float):
    scl = scaletype
    test_case = singleStaircaseController(startVal = start, nReversals = reversals, scale = scl)
    trackvals = []

    while not test_case.testOver:
        
        trackvals.append(test_case.currentValue)

        if test_case.currentValue > true_val:
            test_case.next(True)
        else:
            test_case.next(False)

    print("Test Over")
    print(f"Test took a total of {len(trackvals)} steps")
    print(f"Final Value is {test_case.result}")

    plt.figure()
    plt.plot(trackvals, marker='s')
    plt.grid(True)
    plt.yscale("linear")
    plt.show()


def testMultiStaircase(scaletype: str, numStaircases: int, startVals: list[float], reversals: int, true_val: float):
    scl = scaletype
    test_case = multiStaircaseController(numStaircases, startVals, reversals, scl)

    track_vals1 = []
    track_vals2 = []
    
    while not test_case.allDone:
        
        if test_case.current_staircase == 0:
            track_vals1.append(test_case.currentValue)
        else:
            track_vals2.append(test_case.currentValue)
        
        if test_case.currentValue > true_val:
            test_case.next(True)
        else:
            test_case.next(False)
    
    print("Test Over")
    print(f"S1 took a total of {len(track_vals1)} steps")
    print(f"S2 took a total of {len(track_vals2)} steps")
    print(f"Final Value is {np.mean(test_case.results)}")

    plt.figure()
    plt.plot(track_vals1, marker = 's')
    plt.plot(track_vals2, marker = 's')
    plt.grid(True)
    plt.yscale("linear")
    plt.show()