from tkinter import Image
from skimage.metrics import mean_squared_error
import numpy as np
import numpy as np
import random
import matplotlib.image as mpimg
import numpy as np
import random
from PIL import Image
import numpy as np
import numpy as np
import random

def contar_pixels_3x3(image_path, threshold=128):
    """
    Conta pixels pretos e brancos em uma imagem dividida em 3x3 zonas.
    
    Args:
        image_path (str): Caminho da imagem.
        threshold (int): Limiar para binarização (0-255).
        
    Returns:
        np.array: Vetor de 18 elementos: [pretos_z1, brancos_z1, ..., pretos_z9, brancos_z9]
    """
    # Abre imagem e converte para tons de cinza
    img = Image.open(image_path).convert('L')
    width, height = img.size
    
    zone_width = width // 3
    zone_height = height // 3
    
    features = []
    
    for row in range(3):
        for col in range(3):
            pretos = 0
            brancos = 0
            
            x_start = col * zone_width
            y_start = row * zone_height
            x_end = x_start + zone_width
            y_end = y_start + zone_height
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    pixel = img.getpixel((x, y))
                    if pixel < threshold:
                        pretos += 1
                    else:
                        brancos += 1
            
            features.extend([pretos, brancos])
    
    return np.array(features)

def prepare_template_and_samples_arrays_bw(images_list, num_samples=3):
    """
    Seleciona um template (primeira imagem) e amostras aleatórias,
    retorna todas padronizadas com a mesma dimensão usando padding.
    Funciona apenas para imagens em grayscale (preto e branco).

    Args:
        images_list (list of np.array): Lista de imagens grayscale como arrays.
        num_samples (int): Número de amostras a selecionar.

    Returns:
        tuple: (template, samples) onde ambos são np.array padronizados.
    """
    if len(images_list) < num_samples + 1:
        raise ValueError("Não há imagens suficientes para template e amostras.")

    # Seleciona template e amostras
    template_img = images_list[0]
    sample_imgs = random.sample(images_list[1:], num_samples)

    all_imgs = [template_img] + sample_imgs

    # Descobre altura e largura máximas
    max_height = max(img.shape[0] for img in all_imgs)
    max_width = max(img.shape[1] for img in all_imgs)

    def pad_image(img):
        h, w = img.shape
        # Cria matriz de zeros do tamanho máximo
        canvas = np.zeros((max_height, max_width), dtype=img.dtype)
        # Coloca a imagem original no canto superior esquerdo
        canvas[:h, :w] = img
        return canvas

    # Aplica padding
    images_padded = [pad_image(img) for img in all_imgs]

    template_padded = images_padded[0]
    samples_padded = images_padded[1:]

    return template_padded, samples_padded

# Exemplo de uso
# template, samples = prepare_template_and_samples_arrays_bw(imagens_circles)




# Corr2
def corr2(A, B):
    """
    Calculate the 2D correlation coefficient between two 2D arrays.
    Args:
        A (np.array): First 2D array.
        B (np.array): Second 2D array.
    Returns:
        float: 2D correlation coefficient.
    """
    A = A - np.mean(A)
    B = B - np.mean(B)
    A, B = match_size(A, B)
    return np.sum(A * B) / np.sqrt(np.sum(A**2) * np.sum(B**2))

def match_size(A, B):
    """
    Ajusta os tamanhos de A e B para que tenham as mesmas dimensões.
    Funciona tanto para vetores 1D quanto para matrizes 2D.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Se for 1D, transforma em 2D (coluna)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    rows = min(A.shape[0], B.shape[0])
    cols = min(A.shape[1], B.shape[1])

    return A[:rows, :cols], B[:rows, :cols]

# IMMSE com truncamento
def immse_truncate(X, Y, max_val=520):
    X, Y = match_size(X, Y)
    mse = mean_squared_error(X, Y)
    return mse / (max_val**2) 


# IMMSE
def immse(X, Y):
    return mean_squared_error(X, Y)


def print_scores(results, labeltamplet:str)->None:
    """
    Print the scores of the correlation between the template and the samples.
    Args:
        results (np.array): Array of scores.
        labeltamplet (str): Label of the template.
    Returns:
        None
    """
    i=0
    print(f"Results for template: {labeltamplet}\n")
    for i,scores in enumerate(results):
        if i<3:
            print("Acer Capillipes")
            print(f" Sample {i+1}: {scores:.4f}")
        elif i<6:
            print("Acer Mono")
            print(f" Sample {i-2}: {scores:.4f}")
        else:
            print("Acer Opalus")
            print(f" Sample {i-5}: {scores:.4f}")
        i+=1
    print("\n")