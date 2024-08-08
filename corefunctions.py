from PyQt6.QtWidgets import QApplication
import numpy as np
from PyQt6.QtGui import QGuiApplication
from scipy import signal
from scipy.optimize import least_squares
from PIL import Image
from OpenGL import GL
import ctypes
import matplotlib.pyplot as plt
from math import ceil
import sys

### Screen to Visual Angle Conversion Functions ###

def pix2deg(size_in_pixels, subject_distance, screen_width_in_mm, screen_width_in_pixels):
    """Convert screen pixles to degrees of visual angle
    for given screen parameters and viewing distance.
    
    Parameters:
        size_in_pixels (int): size of stim in pixels.
        subject_distance (int): subject viewing distance from display.
        screen_width_in_mm (float): screen width in millimeters
        screen_width_in_pixels (int): screen width in pixels
        
    Returns:
        float: size of stim in degrees of visual angle
    """
    
    mm_per_pixel = screen_width_in_mm/screen_width_in_pixels
    size_in_mm = size_in_pixels*mm_per_pixel
    visual_angle = np.arctan(size_in_mm/subject_distance)
    visual_angle = np.rad2deg(visual_angle)

    return visual_angle

def deg2pix(size_in_degrees, subject_distance, screen_width_in_mm, screem_width_in_pixels):
    """Convert degrees of visual angle to screen pixles
    for given screen parameters and viewing distance.
    
    Parameters:
        size_in_degrees (float): size of stim in degrees of visual angle.
        subject_distance (int): subject viewing distance from display.
        screen_width_in_mm (float): screen width in millimeters
        screen_width_in_pixels (int): screen width in pixels
        
    Returns:
        int: size of stim in pixels
    """

    mm_per_pixel = screen_width_in_mm/screem_width_in_pixels
    degrees_per_mm = np.rad2deg(np.arctan(1/subject_distance))
    degrees_per_pixel = mm_per_pixel*degrees_per_mm
    pixels_per_degree = 1/degrees_per_pixel
    size_in_pixels = size_in_degrees*pixels_per_degree

    return ceil(size_in_pixels)
    
def getScreenDims(screen = 0, verbose = False):
    """
    Get screen dimensions using QGuiApplication.
    Note: QtWidget.QApplication must be active
    in order for this function to work.
    
    Parameters:
        screen (int): index of screen to query (default is 0)
        
    Returns:
        pixel_size (1x2 array): width and height of screen in pixels
        mm_size (1x2 array) = width and height of screne in millimeters
    """
    
    screens = QGuiApplication.screens()
    pixel_ratio = screens[screen].devicePixelRatio()
    pixel_size = [screens[screen].geometry().width()*pixel_ratio, screens[screen].geometry().height()*pixel_ratio]
    mm_size = [screens[screen].physicalSize().width(), screens[screen].physicalSize().height()]
    
    if verbose:
        print(f"Pixel ratio is {pixel_ratio}")
        print(f"Pixel size is {pixel_size}")
        print(f"Physical size is {mm_size}")

    return pixel_ratio, pixel_size, mm_size

def getNyquist(screen_width_in_mm, screen_width_in_pixels, subject_distance):

    mm_per_pixel = screen_width_in_mm/screen_width_in_pixels
    degrees_per_mm = np.rad2deg(np.arctan(1/subject_distance))
    pixels_per_degree = 1/(mm_per_pixel*degrees_per_mm)
    nyquist = 0.5*pixels_per_degree

    return nyquist


### Numpy 10-bit stimulus creation functions (currently unused but maybe useful later) ###

def make10BitGabor(size, sf = 50, contrast = 0.5, ori = 90, phase = 180, wave = 'sin'):
    """Generate gabor pattern with given sf, contrast, orientation, phase, 
    spatial frequency, and wave pattern as 10-bit unsigned integers packed into
    a 32-bit integer per pixel.

    red = 10 bits
    green = 10 bits
    blue = 10 bits
    alpha = 2 bits

    Parameters:
        size (int): size of the grating (in pixels)
        sf (float): spatial frequency (in pixels per cycle)
        contrast (float): contrast from min to max value (in percent, [0-1])
        ori (int): wave orientation (in degrees, [0-360]
        phase (int): phase of the wave (in degrees, [0-360])
        wave (string): type of wave ('sin' = sine wave, 'sqr' = square wave)

    Returns:
        numpy array: gabor stim of shape (size x size) as unsigned int
    """

    x, y = np.meshgrid(np.arange(size), np.arange(size))
    gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y

    if wave == 'sin':
        grating = np.sin((2*np.pi * gradient) / sf + (phase * np.pi) / 180)
    elif wave == 'sqr':
        grating = signal.square((2 * np.pi * gradient) / sf + (phase * np.pi) / 180)
    else:
        raise NotImplementedError
    
    grating *= contrast
    grating = (grating+1)/2

    grating = grating*1023
    grating = grating.astype(np.uint32)
    alpha = np.ones((size, size), dtype = np.uint32)*1023

    red = grating
    green = grating << 10
    blue = grating << 20
    alpha = alpha << 30

    rgb = red | green | blue | alpha

    return rgb

def make10BitTestRamp(size, start, stop):
    """Generate linear gray test ramp of 10-bit unsigned 
    integers packed into a 32-bit buffer.

    red = 10 bits
    green = 10 bits
    blue = 10 bits
    alpha = 2 bits
    
    Parameters:
        size (int): size of test ramp in pixels
        start (float): starting value between 0 and 1
        stop (float): end value between 0 and 1
        
    Returns:
        list or array: linear gray ramp of size (size x size)
        ranging from (start) to (stop)
    """
    
    gradient = np.linspace(start, stop, size)
    gradient = np.expand_dims(gradient, axis = 1)
    gradient = np.reshape(gradient, newshape = (1,size))
    gradient = np.repeat(gradient, size, axis = 0)
    
    gradient = gradient*1023
    gradient = gradient.astype(np.uint32)
    alpha = np.ones((size, size), dtype = np.uint32)*1023

    red = gradient
    green = gradient << 10
    blue = gradient << 20
    alpha = alpha << 30

    rgb = red | green | blue | alpha
    
    return rgb

def gaussian_filter(size, sigma=0.15):
    """Generate gaussian filter of size (size) with a standard
    deviation of (sigma) percent."""
 
    # Initializing value of x,y as grid of size (size x size)
 
    x, y = np.meshgrid(np.linspace(0, size, size),
                       np.linspace(0, size, size))
    
    # Convert sigma to standard deviation, assign center
    # of image as the mean/peak, and precalculate the divisor
    sd = sigma*size
    center = size//2
    divisor = 2.0*sd**2
 
    # Calculate Gaussian filter
    gauss = np.exp(-(((x-center)**2 + ((y-center)**2))/divisor))

    return gauss

def make8BitGabor(size, sf = 5, contrast = 0.5, ori = 90, phase = 90):
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    ppc = size/sf
    gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y

    grating = np.sin((2*np.pi * gradient) / ppc + (phase * np.pi / 180))

    grating *= contrast

    grating = (grating+1)/2

    gauss = gaussian_filter(size, sigma=0.19)
    gray = 0.5

    R = (grating*gauss) + (gray*(1-gauss))
    G = R
    B = R
    RGB = np.dstack([R, G, B])
    RGB = RGB**(1/2.2)

    return RGB



    

#### CSF data plotting and best fit ####

def lsResiduals(x, sfs, data):
    """Residual function used to calculate best fit for 
    contrast sensitivity function using asymetric parabolic
    function.
    
    Parameters:
        x (list or array): 4-element array of parameters to optimize
        sfs (list or array): spatial frequency values tested
        data (list or array): contrast sensitivity test results
        
    Returns:
        list or array: vector of residuals for least squares fitting
    """
    parabola = csfParabola(sfs, x[0], x[1], x[2], x[3])

    return np.log10(data) - np.log10(parabola)

def csfParabola(spatial_frequencies, peak_sensitivity, peak_frequency, width_l, width_r):
    """Plot contrast sensitivity function as an asymetric parabolic function.
    
    Parameters:
        spatial_frequencies: spatial frequency values across which to compute CSF
        peak_sensitivity: peak contrast sensitivity
        peak_frequency: spatial frequency that coincides with peak_sensitivity
        width_l: width of the left side of the parabola
        width_r: width of the right side of the parabola
        
    Returns:
        y values for asymmetric parabolic function with given parameters
    """

    spatial_frequencies = np.log10(spatial_frequencies)
    peak_sensitivity = np.log10(peak_sensitivity)
    peak_frequency = np.log10(peak_frequency)
    width_l = np.log10(width_l)
    width_r = np.log10(width_r)

    parabola = []
    for value in spatial_frequencies:
        if value < peak_frequency:
            parabola.append(10**(peak_sensitivity - (value - peak_frequency)**2 * (width_l)**2))
        else:
            parabola.append(10**(peak_sensitivity - (value - peak_frequency)**2 * (width_r)**2))

    return parabola

def csfBestFit(best_fit_xvals, data_xvals, data):
    """Calculate best fit for contrast sensivitify data
    using asymetric parabolic function and least squares.

    Parameters:
        best_fit_xvals (list or array): x values used to calculate best fit values
        data_xvals (list or array): x values for your data
        data (list or array): data from your experiment
    
    Returns:
        bestFit (list or array): y values of best fit line
    """
    # Generate best guess for parameters using the input data

    peak_sensitivity_guess = 150
    peak_frequency_guess = 3.5
    width_l_guess = 5.0
    width_r_guess = 20.0
    x0 = [peak_sensitivity_guess, peak_frequency_guess, width_l_guess, width_r_guess]
    ls_bounds = [[5, 0.1, 1.2, 1.2], [500, 32, 500.0, 500.0]]

    # Use scipy least squares to calculate best fit parameters
    ls_results = least_squares(lsResiduals, x0, args = (data_xvals, data), method='trf', verbose=False, bounds=ls_bounds)

    # Calculate best fit values using best fit parameters and a given set of x values
    bestFit = csfParabola(best_fit_xvals, ls_results.x[0], ls_results.x[1], ls_results.x[2], ls_results.x[3])

    return bestFit



### OpenGL Helper Functions ###

def genTextureFromImage(filename):
    """Generate and bind an OpenGL texture from an image by filename.
    Returns the texture ID"""

    image = Image.open(filename)
    image.convert("RGBA")
    texture = GL.GLuint()
    GL.glGenTextures(1, ctypes.byref(texture))
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, image.width,
                    image.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, 
                    image.tobytes())

    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    return texture
    
def genQuadWithTextureCoords(quad_width, quad_height, screen_width, screen_height, x_offset = 0, y_offset = 0):
    """Generate vertices for an OpenGL 'quad' (a rectangle made out of two triangles) with
    a given width, height, x offset and y offset that also has a texture bound to its corners."""

    width_percent = quad_width/screen_width
    height_percent = quad_height/screen_height
    x_offset_percent = (x_offset/screen_width)*2
    y_offset_percent = (y_offset/screen_height)*2
    
    vertices = (0.0-width_percent+x_offset_percent, 0.0-height_percent+y_offset_percent, 0.0, 0.0, 1.0,
            0.0+width_percent+x_offset_percent, 0.0-height_percent+y_offset_percent, 0.0, 1.0, 1.0,
            0.0-width_percent+x_offset_percent, 0.0+height_percent+y_offset_percent, 0.0, 0.0, 0.0,
            
            0.0-width_percent+x_offset_percent, 0.0+height_percent+y_offset_percent, 0.0, 0.0, 0.0,
            0.0+width_percent+x_offset_percent, 0.0+height_percent+y_offset_percent, 0.0, 1.0, 0.0,
            0.0+width_percent+x_offset_percent, 0.0-height_percent+y_offset_percent, 0.0, 1.0, 1.0)
    
    vertices = np.asarray(vertices, dtype = np.float32)
    vertex_count = len(vertices)//5

    return vertices, vertex_count

def createShaderProgram(vertex_filename, fragment_filename):
    """Create vertex and fragment shaders, compile them, and link them
    into a shader program.
    
    Returns linked shader program."""

    with open(vertex_filename, 'r') as f:
        vertex_src = f.readlines()

    with open(fragment_filename, 'r') as f:
        fragment_src = f.readlines()

    vertex = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    fragment = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(vertex, vertex_src)
    GL.glShaderSource(fragment, fragment_src)
    
    success = GL.GLint()
    GL.glCompileShader(vertex)
    GL.glGetShaderiv(vertex, GL.GL_COMPILE_STATUS, ctypes.byref(success))

    if not success:
        raise Exception("ERROR: Vertex Shader Failed to Compile")

    GL.glCompileShader(fragment)
    GL.glGetShaderiv(fragment, GL.GL_COMPILE_STATUS, ctypes.byref(success))

    if not success:
        raise Exception("ERROR: Fragment Shader Failed to Compile")
    
    shader_program = GL.glCreateProgram()
    GL.glAttachShader(shader_program, vertex)
    GL.glAttachShader(shader_program, fragment)
    GL.glLinkProgram(shader_program)
    GL.glGetProgramiv(shader_program, GL.GL_LINK_STATUS, ctypes.byref(success))

    if not success:
        raise Exception("ERROR: Shader Program Failed to Link")
    
    GL.glDeleteShader(vertex)
    GL.glDeleteShader(fragment)
    
    return shader_program

def genVAOandVBOWithTextureCoords(vertices):


    """Generate a Vertex Array Object and Vertex Buffer Object from
    a given set of vertices that include vertex position coordinates
    and texture coordinates"""

    vao = GL.GLuint()
    GL.glGenVertexArrays(1, ctypes.byref(vao))
    GL.glBindVertexArray(vao)

    vbo = GL.GLuint()
    GL.glGenBuffers(1, ctypes.byref(vbo))
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes,
                        vertices, GL.GL_STATIC_DRAW)
    
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)

    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(12))
    GL.glEnableVertexAttribArray(1)

    return vao, vbo

