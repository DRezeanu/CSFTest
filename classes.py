from functions import (deg2pix, getScreenDims, createShaderProgram,
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

## OpenGL Windows and Stimulus Classes ##

class GLImageStim:

    """OpenGL image stimulus class with given size, location and, of course, the actual image to
    be used."""

    def __init__(self, filename, width, height, x_offset = 0, y_offset = 0, subject_distance = 1000):
        
        # Get screen dimentions
        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        # Specific to mac screens, OpenGL seems to ignore pixel_ratio so I'm discounting it here
        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        # Initialize basic paramters like the size of the stim, the x and y offset
        # and the filename of the image to be used.
        self.filename = filename
        self.width = width
        self.height = height
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.quad_width = deg2pix(self.width, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.quad_height = deg2pix(self.height, self.subject_distance, physical_dims[0], pixel_dims[0])

        # Generate OpenGL vertices and texture using custom functions
        self.vertices, self.vertex_count = genQuadWithTextureCoords(self.quad_width, self.quad_height, pixel_dims[0], 
                                                              pixel_dims[1], self.x_offset, self.y_offset)
        
        self.texture = genTextureFromImage(self.filename)
        
        # Create vao and vbo to hold vertex and image data
        self.vao, self.vbo = genVAOandVBOWithTextureCoords(self.vertices)

        # Create shader program to actually draw the vertices and pixels
        self.shader_program = createShaderProgram("Shaders/vertex_shader.txt", "Shaders/image_frag_shader.txt")
        self.tex_uniform = GL.glGetUniformLocation(self.shader_program, "imageTexture")

        # Use the program and set the texture to the image we selected
        GL.glUseProgram(self.shader_program)
        GL.glUniform1i(self.tex_uniform, 0)

    def use(self):
        """Method for using the generated image texture by binding it, binding the VAO
        Using the program. Setting the texture uniform, and drawing the array."""

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glBindVertexArray(self.vao)
        GL.glUseProgram(self.shader_program)
        GL.glUniform1i(self.tex_uniform, 0)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        """Method for destroying the generated image texture. Called on window close
        to free up memory. Deletes VBO, deletes VAO, and deletes shader program."""

        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)


class GLGratingStim:

    def __init__(self, size = 4, x_offset = 0, y_offset = 0, subject_distance = 1000,
                 sf = 4, ori = 0, contrast = 0.1, phase = 0, sd = 0.15, wave = 'sin'):
        
        """OpenGL gabor stimulus class with given size, location, spatial frequency, orientation
        contrast, and phase."""

        # Get  screen dimensions
        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        # Specific to mac screens, OpenGL seems to ignore pixel_ratio so I'm discounting it here
        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        # Initialize basic parameters like the wave type (sin or square), size, spatial frequency
        # contrast, phase, orientation, x and y offset, etc.
        self.wave = wave
        self.size = size
        self.sf = sf
        self.subject_distance = subject_distance
        self.x_offset = deg2pix(x_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.y_offset = deg2pix(y_offset, self.subject_distance, physical_dims[0], pixel_dims[0])
        self.ori = ori
        self.contrast = contrast
        self.phase = phase
        self.quad_size = deg2pix(self.size, self.subject_distance, physical_dims[0], pixel_dims[0])

        # Generate OpenGL vertices with texture coordinates that will be used by the shader
        # to draw the stimulus correctly
        self.vertices, self.vertex_count = genQuadWithTextureCoords(quad_width = self.quad_size,
                                                               quad_height = self.quad_size,
                                                               screen_width = pixel_dims[0],
                                                               screen_height = pixel_dims[1],
                                                               x_offset = self.x_offset,
                                                               y_offset = self.y_offset)
        
        # Create vao and vbo to hold vertex and pixel data
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

        # Convert spatial frequency from cycles per degree to cycles per stimulus, as this
        # is how the shader expects to receive this input
        self.sf *= self.size

        # Use the shader program and set all of the uniforms to their initial values
        GL.glUseProgram(self.shader_program)
        GL.glUniform1f(self.u_orientation, self.ori)
        GL.glUniform1f(self.u_contrast, self.contrast)
        GL.glUniform1f(self.u_sf, self.sf)
        GL.glUniform1f(self.u_phase, self.phase)

    def use(self):
        """Method for using the gabor stimulus by calling the correct shader program
        Binding the correct vertex array, and drawing the results"""

        GL.glUseProgram(self.shader_program)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        """Method for destroying the gabor stimulus. To be called on window close.
        This deletes the VAO, the VBO and the shader program to free up memory."""

        GL.glDeleteBuffers(1, ctypes.byref(self.vbo))
        GL.glDeleteVertexArrays(1, ctypes.byref(self.vao))
        GL.glDeleteProgram(self.shader_program)


class GL_CSFDemoWindow(QOpenGLWindow):

    """Subclass of PyQt6 QOpenGLWindow class that includes all of the timing and controllers
    necessary to run the "Demo" part of the program."""

    def __init__(self, subject_distance, stim_size = 4, eccentricity = 4, parent = None):

        # Initialize base variables for stim size, eccentricity, and subject distance
        super(GL_CSFDemoWindow, self).__init__(parent)
        self.subject_distance = subject_distance
        self.stim_size = stim_size
        self.eccentricity = eccentricity

        # Load the beep that plays to indicate a key press was registered
        self.confirmation_beep = QSoundEffect()
        self.confirmation_beep.setSource(QUrl.fromLocalFile("Assets/confirmation.wav"))

    def initializeGL(self) -> None:

        #Initialize the boolean values that will determine what is being shown on screen
        self.page1 = True
        self.page_correct = False
        self.page_wrong = False
        self.show_fixation = False

        # Initialize boolean value that will determine if relevant keys are active
        self.YN_active = False
        self.space_active = True
        self.arrows_active = False

        # Set the background of the screen to linear 50% gray
        GL.glClearColor(0.5**(1/2.2), 0.5**(1/2.2), 0.5**(1/2.2), 1.0)

        # get screen dimensions
        pixel_ratio, pixel_dims, physical_dims = getScreenDims()

        # Specific to mac screens, OpenGL seems to ignore pixel_ratio so I'm discounting it here
        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        # Convert screen pixel dimensions to degrees of visual angle
        degree_dims = [pix2deg(pixel_dims[0], self.subject_distance, physical_dims[0], pixel_dims[0]), 
                       pix2deg(pixel_dims[1], self.subject_distance, physical_dims[0], pixel_dims[0])]

        # If the screen is only capable of 8-bits per color channel, warn the user that they may not
        # get accurate results at low contrasts.
        if self.context().format().redBufferSize() == 8:
            QMessageBox.critical(None, "Color Depth Warning", """Currently running in 8-bit color mode.
                                Maximum contrast limited!""", QMessageBox.StandardButton.Ok)

        # Create custom controller class to handle timing
        self.demoController = DemoController(num_stims=4)
        
        # Generate stimulus gabors as OpenGL textures
        self.top_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = self.eccentricity)
        self.bottom_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = -self.eccentricity)
        self.right_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = self.eccentricity, y_offset = 0)
        self.left_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = -self.eccentricity, y_offset = 0)

        # Generate fixation point and various message screens as OpenGL image textures
        self.fixation = GLImageStim("Assets/fixation_dot.png", width = 0.5, height = 0.5, subject_distance = self.subject_distance)
        self.explanation = GLImageStim("Assets/explanation_window.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.press_space = GLImageStim("Assets/begin_demo.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.correct_message = GLImageStim("Assets/correct.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.wrong_message = GLImageStim("Assets/wrong.png", width = degree_dims[1], height = degree_dims[1], subject_distance=self.subject_distance)

        # Sync screen repaint signal to the vertical refresh rate
        self.frameSwapped.connect(self.update)

        # Create a value that holds the correct answer (1 is top, 2 is right, 3 is bottom, and 4 is left)
        self.correct = 0

        # Enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Set display to full screen
        self.showFullScreen()
    
    def resizeGL(self, w: int, h: int) -> None:
        """Boilerplate function that handles any resizing of the viewport.
        Shouldn't happen but including just in case."""
        return super().resizeGL(w, h)
    
    def paintGL(self) -> None:
        # Clear the color buffer bit to reset the draw buffer
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        # Draw the correct stimulus or message depending on which control variables
        # are set to "True"
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
        """Method for handling key press events"""

        # Make context current so this can appropriately set all OpenGL variables
        self.makeCurrent()

        # Allow user to close program with ESC key at any time
        if event.key() == Qt.Key.Key_Escape:
            self.close()

        # What happens when you press Space (and Space key is active)
        if event.key() == Qt.Key.Key_Space:
            if self.space_active:
                self.page1 = False
                self.show_fixation = True
                self.shuffleAttributes()
                self.demoController.next()
                self.space_active = False
                self.arrows_active = True

        # What happesn when you press "Y" (and Y/N keys are active)
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

        # What happens when you press N (and Y/N keys are active)
        if event.key() == Qt.Key.Key_N:
            if self.YN_active:
                self.close()

        # What happens when you press arrow keys (and arrow keys are active)
        # Changes based on whether or not the stimulus was in that location
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

    def shuffleAttributes(self):
        """Method for shuffling the attributes of the gabor stimuli randomly
        for the purposes of demoing different contrast, spatial frequencies, 
        orientations, and phases for each run of the demo."""

        # shuffle contrast and set all uniforms to the new value
        contrast = random.choice([0.08, 0.16, 0.32, 0.64])

        GL.glUseProgram(self.top_gabor.shader_program)
        GL.glUniform1f(self.top_gabor.u_contrast, contrast)

        GL.glUseProgram(self.bottom_gabor.shader_program)
        GL.glUniform1f(self.bottom_gabor.u_contrast, contrast)

        GL.glUseProgram(self.right_gabor.shader_program)
        GL.glUniform1f(self.right_gabor.u_contrast, contrast)

        GL.glUseProgram(self.left_gabor.shader_program)
        GL.glUniform1f(self.left_gabor.u_contrast, contrast)


        # shuffle orientation and set all uniforms to the new value
        ori = random.choice([0, 45, 90, 135])

        GL.glUseProgram(self.top_gabor.shader_program)
        GL.glUniform1f(self.top_gabor.u_orientation, ori)

        GL.glUseProgram(self.bottom_gabor.shader_program)
        GL.glUniform1f(self.bottom_gabor.u_orientation, ori)

        GL.glUseProgram(self.right_gabor.shader_program)
        GL.glUniform1f(self.right_gabor.u_orientation, ori)

        GL.glUseProgram(self.left_gabor.shader_program)
        GL.glUniform1f(self.left_gabor.u_orientation, ori)


        # shuffle sf, convert to cycles per stimulus, and set all
        # uniforms to the new value
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
        """Overrite basic close function to include the destroy method on all of the stimuli
        that we created, freeing up memory."""
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

    """Subclass of PyQt6 QOpenGLWindow class that includes all of the timing and controllers
    necessary to run the the main CSF testing paradigm program."""

    # Custom PyQt signal to indicate that the test has finished and deliver the results
    # as a dictionary to the main window for plotting
    finished=pyqtSignal(dict)

    # initialize basic variables like subject distance, stim size, and eccentricity
    def __init__(self, subject_distance, stim_duration = 200, stim_size = 4, eccentricity = 4, parent=None):
        super(GL_CSFTestWindow, self).__init__(parent)
        self.subject_distance = subject_distance
        self.eccentricity = eccentricity
        self.stim_size = stim_size
        self.duration = stim_duration
        self.confirmation_beep = QSoundEffect()
        self.confirmation_beep.setSource(QUrl.fromLocalFile("Assets/confirmation.wav"))

    def initializeGL(self) -> None:
        
        # Initialize boolean values that determine if space or arrow keys are active
        self.spaceActive = True
        self.arrowsActive = False

        # Initialize the boolean values that will determine what is being shown on screen
        self.first_page = True
        self.show_stim = False
        self.show_fixation = False

        # set background color of main window to linearized 50% gray
        GL.glClearColor(0.5**(1/2.2), 0.5**(1/2.2), 0.5**(1/2.2), 1.0)

        # get screen dims and nyquist frequency of this display given this subject distance
        pixel_ratio, pixel_dims, physical_dims = getScreenDims()
        nyquist = getNyquist(physical_dims[0], pixel_dims[0], self.subject_distance)

        # Specific to mac screens, OpenGL seems to ignore pixel_ratio so I'm discounting it here
        if pixel_ratio > 1:
            pixel_dims[0] /= pixel_ratio
            pixel_dims[1] /= pixel_ratio

        # Convert screen pixel dimensions to degrees of visual angle
        degree_dims = [pix2deg(pixel_dims[0], self.subject_distance, physical_dims[0], pixel_dims[0]), 
                       pix2deg(pixel_dims[1], self.subject_distance, physical_dims[0], pixel_dims[0])]

        # If the screen is only capable of 8-bits per color channel, warn the user that they may not
        # get accurate results at low contrasts.
        if self.context().format().redBufferSize() == 8:
            QMessageBox.critical(None, "Color Depth Warning", """Currently running in 8-bit color mode.
                                Maximum contrast limited!""", QMessageBox.StandardButton.Ok)
        
        # Generate gabor stimuli as OpenGL textures
        self.top_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = self.eccentricity)
        self.bottom_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = 0, y_offset = -self.eccentricity)
        self.right_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = self.eccentricity, y_offset = 0)
        self.left_gabor = GLGratingStim(size = self.stim_size, subject_distance = self.subject_distance, sf = 1, contrast = 0.5, phase = 0, ori = 0, x_offset = -self.eccentricity, y_offset = 0)

        # Generate fixation dot and various message screens as OpenGL image textures
        self.fixation = GLImageStim("Assets/fixation_dot.png", width = 0.5, height = 0.5, subject_distance = self.subject_distance)
        self.explanation = GLImageStim("Assets/start_window.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.break_time = GLImageStim("Assets/break_time.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)
        self.test_over = GLImageStim("Assets/all_done.png", width = degree_dims[1], height = degree_dims[1], subject_distance = self.subject_distance)

        # Create trial handler and controller classes
        self.trialHandler = TrialHandler(stim_size = self.stim_size)
        
        # Throw an error if the maximum spatial frequency being tested is higher than the nyquist limit
        if self.trialHandler.sfMax > nyquist:
            raise ValueError("Max spatial frequency exceeds nyquist limit for this display and disatnce")
        
        self.displayHandler = DisplayHandler(num_stims = 4, stim_duration = self.duration, pre_stim_interval=1500)

        # Add all relevant shaders and uniforms to lists for easy access
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

        # Enable alpha blending
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Set display to full screen
        self.showFullScreen()

    def resizeGL(self, w: int, h: int) -> None:
        """Boilerplate function that handles any resizing of the viewport.
        Shouldn't happen but including just in case."""

        return super().resizeGL(w, h)
    
    def paintGL(self) -> None:

        # Clear the color buffer bit to reset the draw buffer
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # Draw the correct stimulus or message depending on which control variables
        # are set to "True"
        if self.trialHandler.trialOver:
            self.break_time.use()
            self.spaceActive = True
            self.arrowsActive = False
        elif self.trialHandler.testOver:
            self.test_over.use()
            self.spaceActive = True
            self.arrowsActive = False
        else:
            if self.first_page:
                self.explanation.use()

            if self.show_fixation:
                self.fixation.use()

            if self.displayHandler.trigger[0]:
                self.top_gabor.use()
            
            if self.displayHandler.trigger[1]:
                self.right_gabor.use()
            
            if self.displayHandler.trigger[2]:
                self.bottom_gabor.use()

            if self.displayHandler.trigger[3]:
                self.left_gabor.use()
        
    def keyPressEvent(self, event):
        """Method for handling key press events"""

        # Allow user to exit program at any time by pressing ESC
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        
        # What happens when you press the Space Key (using control variables)
        if event.key() == Qt.Key.Key_Space:
            if self.spaceActive:
                if self.first_page:
                    self.first_page = False
                    self.show_fixation = True
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.current_stim_params
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
                    self.displayHandler.showStim(stim_params, shader, uniforms)
                    self.arrowsActive = True
                    self.spaceActive = False
                    time.sleep(0.5)
                    

        # What happens when you press the arrow keys (using control variables)
        if event.key() == Qt.Key.Key_Up:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 0:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.arrowsActive = True


        if event.key() == Qt.Key.Key_Right:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 1:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.arrowsActive = True

        if event.key() == Qt.Key.Key_Down:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 2:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.arrowsActive = True

        if event.key() == Qt.Key.Key_Left:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                if self.displayHandler.currentStim == 3:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(1)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                else:
                    shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                    stim_params = self.trialHandler.nextStim(0)
                    if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                        self.displayHandler.showStim(stim_params, shader, uniforms)
                self.arrowsActive = True
        
        # Z key is used as a signal that user didn't see anything and would be guessing.
        # May make results more accurate but it means the test is no longer a true alternative
        # forced choice, so leaving it up to the experimenter on whether or not they reveal this
        # feature to the user
        if event.key() == Qt.Key.Key_Z:
            if self.arrowsActive:
                self.confirmation_beep.play()
                self.arrowsActive = False
                shader, uniforms = self.displayHandler.pickStim(self.gabor_shaders, self.gabor_uniforms)
                stim_params = self.trialHandler.nextStim(0)
                if not self.trialHandler.trialOver and not self.trialHandler.testOver:
                    self.displayHandler.showStim(stim_params, shader, uniforms)
                self.arrowsActive = True

    def close(self):
        """Overrite basic close function to include the destroy method on all of the stimuli
        that we created, freeing up memory."""
        self.top_gabor.destroy()
        self.left_gabor.destroy()
        self.right_gabor.destroy()
        self.bottom_gabor.destroy()
        self.fixation.destroy()
        self.explanation.destroy()
        self.break_time.destroy()
        super().close()



## Trial and Display Controller Classes ##

class DemoController:
    
    """Custom controller class that sets timing and display of stimuli in "Demo" mode."""

    def __init__(self, num_stims = 4, stim_duration = 100, countdown_time = 2000):
        
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

    """Custom staircase controller that takes a start value, a list of step sizes
    and a number of reversals (as an integer) and generates an adaptive staircase
    paradigm used by calling the .next() method"""

    def __init__(self, startVal: float, stepSizes: list[float], nReversals: int):
        
        self._startVal = startVal
        self._stepSizes = stepSizes
        self._nReversals = nReversals
        self._countReversals = 0
        self._reversalValues = []
        self._currentStep = self._stepSizes[0]
        self._isReversal = False
        self.testOver = False
        self.result = None
        
        self._previousAnswer = None
        self.currentValue = self._startVal

    def next(self, userInput: bool):
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
        
        if userInput:
            self.currentValue = np.max([self.currentValue-self._currentStep, 0])
        else:
            self.currentValue = np.min([self.currentValue+self._currentStep, 1])

        return self.currentValue


    def decreaseStepSize(self):
        
        idx = self._stepSizes.index(self._currentStep)

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

    """Custom Multi-staircase controller class that basically mixes however many staircases you want to use
    and randomly selects between them."""

    def __init__(self, numStaircases: int, startVals: list[float], stepSizes: list[float], nReversals: int):
        
        self.numStaircases = numStaircases
        self.stepSizes = stepSizes
        self.startVals = startVals
        self.nReversals = nReversals

        if len(startVals) != numStaircases:
            raise ValueError("You must have the same number of start values as staircases")
        
        # if number of step sizes doesn't match the number of reversals specified, truncate the later to match
        if len(stepSizes) != nReversals:
            nReversals = len(stepSizes)        

        # Initialize all staircases
        self.staircases: list[singleStaircaseController] = []
        for i in range(numStaircases):
            self.staircases.append(singleStaircaseController(startVals[i], stepSizes, nReversals))

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

    """Custom trial handler class that creates and runs multiple staircases to completion,
    spitting out new stimulus parameters and control variables (e.g. is the trial over? or is 
    the test over?) as appropriate."""

    def __init__(self, stim_size, sfMin = 0.5, sfMax = 32, numTrials: int = 7, numStaircases: int = 2,
                 stepSizes: list[float] = [0.05, 0.025, 0.015, 0.01, 0.005, 0.0025, 0.0015], 
                 nReversals: int = 7):
        
        self.sfMin = sfMin
        self.sfMax = sfMax
        self.stim_size = stim_size
        self.numTrials = numTrials
        self.numStaircases = numStaircases
        self.stepSizes = stepSizes
        self.nReversals = nReversals
        self.currentTrial = 0
        self.results={}
        self.phase = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        self.ori = np.random.choice([0, 45, 90, 135])
        self.testOver = False
        self.trialOver = False

        # Generate list of spatial frequencies to test
        self.SFs = np.geomspace(sfMin, sfMax, self.numTrials)

        # Shuffle their order in place
        np.random.shuffle(self.SFs)

        # generate the expected values given an average CSF, and set start values for staircases accordingly
        expected_values = csfParabola(self.SFs, 200, 3, 3.5, 20)
        self.startVals = np.empty((self.numTrials, self.numStaircases))

        for i in range(len(expected_values)):
            # self.startVals[i,:] = [np.max([expected_values[i]*2,1]), np.max([expected_values[i]/2,1])]
            self.startVals[i,:] = [1/0.005, 1/0.6]

        self.startVals = 1/self.startVals

        self.staircaseHandler = self.genStaircase(self.currentTrial)

        # Set current stim parameters
        self.current_stim_params = [self.SFs[self.currentTrial]*self.stim_size, self.ori, self.phase, self.staircaseHandler.currentValue]
    
    
    def nextStim(self, userInput):
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
            self.current_stim_params[1] = np.random.choice([0, 45, 90, 135])
            self.current_stim_params[2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            self.current_stim_params[3] = newVal
        else:
            newVal = self.staircaseHandler.next(userInput)
            self.current_stim_params[1] = np.random.choice([0, 45, 90, 135])
            self.current_stim_params[2] = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            self.current_stim_params[3] = newVal
        
        return self.current_stim_params


    def genStaircase(self, currentTrial):

        staircase = multiStaircaseController(self.numStaircases,
                                                         self.startVals[currentTrial,:],
                                                         self.stepSizes,
                                                         self.nReversals)
        return staircase


class DisplayHandler:

    """Custom display handler class that determines what is being shown on the screen
    at any given time during the CSF test."""

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

    def pickStim(self, shaders, uniformArray):
        self.currentStim = np.random.choice(np.arange(0, self.numStims))
        shader = shaders[self.currentStim]
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

    """Custom plotting subclass that generates the Matplotlib plot shown on the main window
    is used to plot the results at the end of the test, and saves the output as a PNG file.
    This class is initialized with some placeholder results so it's not just an empty axis."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(ResultsPlot, self).__init__(fig)
        self.setPlaceholderResults()

    def setPlaceholderResults(self):
        xvals = np.geomspace(0.5, 32, 50)
        sampleSFs = np.asarray([0.5, 1, 2, 4, 8, 16, 32])
        sampleData = np.asarray([45, 100, 125, 150, 125, 30, 2])

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