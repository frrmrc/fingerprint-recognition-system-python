import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import thin
from pathlib import Path
import uuid 
import storage as s

class fingerprint(object): # fingeprint (?)

    def __init__(self,filepath):
        self.id= str(uuid.uuid4())
        self.path=filepath
        self.resized=None #img --> array numpy
        self.img_blur=None #img --> array numpy
        self.img_morf=None #img --> array numpy
        self.img_skel=None #img --> array numpy
        self.polished=None #img --> array numpy
        self.minutiae=None #df pandas
        self.orientation_map=None #file numpy
        self.steps_img=None #img--> matplotlib figure

    def rotate_fp(self,degrees): #rotates the image using function rotate()
        img= cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.resized=rotate(img,degrees)# Warning! Is stored in self.resized, but doesn't go through resize()

    def resize(self): #Resizes image to 241*298 px
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        target_w, target_h = (241, 298)
        if (w,h)==(target_w, target_h):
            self.resized=img
            return
        scale = min(target_w / w, target_h / h) # Calculates the scale factor to maintain proportions
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) #Resizes the image while 
        #Calculates the margins to center the image.        
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        # Adds borders
        self.resized= cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    def polish(self): #Returns the fingerprint's skeleton
        img_norm = cv2.normalize(self.resized, None, 0, 255, cv2.NORM_MINMAX)
        self.img_blur = cv2.bilateralFilter(img_norm, d=7, sigmaColor=45, sigmaSpace=45) # Bilateral filter 
        img_bin = cv2.adaptiveThreshold(self.img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1) # Adaptive treshold
        img_bin_bool = img_bin == 0 
        kernel = np.ones((1,1), np.uint8)  
        self.img_morf= cv2.morphologyEx(img_bin_bool.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        self.img_skel = thin(self.img_morf) #Applying thickness reduction (thin).
        self.polished = (self.img_skel * 255).astype(np.uint8) #Conversion to uint8        
    
    def find_minutiae(self): #Extract minutiae points from the image and saves to pd dataframe
        self.minutiae = []
        rows, cols = self.polished.shape
        for y in range(3, rows -3):
            for x in range(3, cols -3):
                if self.polished[y, x] == 255: 
                    neighbors = [
                        self.polished[y-1, x-1], self.polished[y-1, x], self.polished[y-1, x+1],
                        self.polished[y, x-1],                          self.polished[y, x+1],
                        self.polished[y+1, x-1], self.polished[y+1, x], self.polished[y+1, x+1]
                    ]
                    white_neighbors = sum(n == 255 for n in neighbors)
                    if white_neighbors == 1:
                        self.minutiae.append((x, y, "end"))  
                    elif white_neighbors >= 3:
                        self.minutiae.append((x, y, "bifurcation"))  
        self.minutiae = pd.DataFrame(self.minutiae, columns=["x", "y", "tipo"])
        
    def gen_orientation_map(self): #Generates Orientations map
        #Calculates the gradient of the image using the Sobel operator 
        Gx = cv2.Sobel(self.polished, cv2.CV_64F, 1, 0, ksize=3)  # orizontal 
        Gy = cv2.Sobel(self.polished, cv2.CV_64F, 0, 1, ksize=3)  # vertical
        Gx2 = Gx**2
        Gy2 = Gy**2
        GxGy = Gx * Gy
        Sx2 = cv2.GaussianBlur(Gx2, (5, 5), 1)
        Sy2 = cv2.GaussianBlur(Gy2, (5, 5), 1)
        Sxy = cv2.GaussianBlur(GxGy, (5, 5), 1)
        theta = 0.5 * np.arctan2(2 * Sxy, Sx2 - Sy2)  #Converts from radians to degrees
        self.orientation_map = np.degrees(theta) % 180
    
    def lay(self): # Combines the necessary steps to start the analysis
        self.resize()
        self.polish()
        self.gen_orientation_map()
        self.find_minutiae()
    
    def visual_steps(self): #Visualizes image preparation steps
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(self.resized, cmap='gray')
        ax[0].set_title("Originale")
        ax[1].imshow(self.img_blur, cmap='gray')
        ax[1].set_title("Filtro Bilaterale")
        ax[2].imshow(self.img_morf, cmap='gray')
        ax[2].set_title("Binarizzazione")
        ax[3].imshow(self.img_skel, cmap='gray')
        ax[3].set_title("Scheletrizzazione")
        for a in ax:
            a.axis("off")
        self.steps_img=fig


class search(object): #fingerprint, but when you need to look for it

    def __init__(self,fingerprint):
        self.filepath=fingerprint.path #filepath
        self.fingerprint_object=fingerprint #fingerprint class object
        self.minutiae=None #pd dataframe
        self.max_degrees=None #int
        self.step=None #int
        self.top5maps=None #pd dataframe
        self.top5minutiae=None #pd dataframe
        self.matches=None #list
        self.match_img=None #img --> matplotlib figure
        
    def find_similar_maps(self,maxdegrees=15, step=1): #Returns a list of tuples of the 5 most similar fingerprints with name, error, and rotation degrees
        wanted=self.fingerprint_object.orientation_map 
        similarità=[]   
        count=0
        mappe_dir = "orientationmaps"  
        for file in os.listdir(mappe_dir): #Goes through all the .npy files saved 
            mappa_dataset = np.load(os.path.join(mappe_dir, file))
            min_errore = float("inf")
            migliore_rotazione = None
            for degrees in range(-maxdegrees, maxdegrees, step):  
                mappa_ruotata = rotate(mappa_dataset, degrees)
                differenza = np.abs(wanted - mappa_ruotata)  
                errore_medio = np.mean(differenza)
                count+=1
                if errore_medio < min_errore:
                    min_errore = errore_medio
                    migliore_rotazione = degrees
            similarità.append((file[:-4], min_errore, migliore_rotazione)) # Saves the most similar rotation
        similarità.sort(key=lambda x: x[1])  # Orders by error
        self.top5maps=similarità[:5]
        return count #count of orientation maps checked
   
    def rotate_extract(self): #returns a pd dataframe with minutiaes of the 5 best matchese
        filenames, errore, bestrotation= zip(*self.top5maps)
        strikes=[]
        for file,rotation in zip(filenames, bestrotation):
            path=os.path.join('cron_upload',file+'.bmp')
            impronta=fingerprint(path)
            if rotation!=0:
                impronta.rotate_fp(rotation)
                impronta.polish()
                impronta.find_minutiae()
            else:
                impronta.minutiae=pd.read_csv(os.path.join('minutiae',file+'.csv'))
            for x, y, tipo in  zip(impronta.minutiae['x'], impronta.minutiae['y'], impronta.minutiae['tipo']):
                strikes.append((file, x, y, tipo))
        self.top5minutiae = pd.DataFrame(strikes, columns=["file", "x", "y", "tipo"])  

    def best_match_minutiae(self):# Finds best match based on orientation maps and minutiaes
        t = 10  # Maximum distance between minutiaes to consider the match valid
        w1, w2 = 0.4, 0.6  #Wheights the scores: w1 for orientation maps match and w2 for minutiaes match
        tipo_peso = 1.5  #greater point for minutiaes of the same time
        top5=self.top5maps
        matches_list=[]
        wanted_minutiae=self.fingerprint_object.minutiae
        # Min e max error per normalizzare
        errori = [r[1] for r in top5] #Exctracts error score from top 5 maps
        min_error, max_error = min(errori), max(errori)
        for filename, errore, degrees in top5: #iterates on fingerprint
            normalized_error = (errore - min_error) / (max_error - min_error) # normalizes error from orientation maps
            each=self.top5minutiae[self.top5minutiae['file']==filename][["x", "y", "tipo"]].values
            matches = 0
            for _, row in wanted_minutiae.iterrows(): #iterates on wanted fingerprint minutiae
                x1, y1, tipo1 = row['x'], row['y'], row['tipo']
                for x2, y2, tipo2 in each: #for each minutiae from wanted fingerprint ones, iterates on all the other minutiaes of the fingerprint being analyzed
                    distanza = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if distanza < t:  #if the distance is ok
                        if tipo1 == tipo2:
                            matches += tipo_peso  #  If it's of the same type: greater score
                        else:
                            matches += tipo_peso/2  # If it's not: smaller score
            matches_list.append(matches)
        max_matches = max(matches_list, default=1)  #Avoiding divisions by zero   
        self.matches = []
        for (filename, errore, degrees), matches in zip(top5, matches_list): #calculating the final score
            normalized_error= (errore - min_error) / (max_error - min_error) 
            normalized_minutiae_score=matches / max_matches 
            final_score = w1 * (1 - normalized_error) + w2 * normalized_minutiae_score  #final score
            self.matches.append((filename, final_score))
        self.matches.sort(key=lambda x: x[1], reverse=True) #orders by best score
    
    def visual_match(self): #Plot with orientation map and minutiae of wanted and best match
        img_best_match=cv2.imread(os.path.join('cron_upload',self.matches[0][0]+'.bmp'), cv2.IMREAD_GRAYSCALE)
        minutiae_best_match=pd.read_csv(os.path.join('minutiae',self.matches[0][0]+'.csv'))
        plot_minutiae_search=plot_minutiae(self.fingerprint_object.resized,self.fingerprint_object.minutiae)
        plot_minutiae_match=plot_minutiae(img_best_match,minutiae_best_match) 
        plot_om_search=self.fingerprint_object.orientation_map
        plot_om_match=np.load(os.path.join('orientationmaps',self.matches[0][0]+'.npy'))
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 5, width_ratios=[0.05, 1, 1, 1, 1])  
        ax0 = fig.add_subplot(gs[0])  
        ax1 = fig.add_subplot(gs[1])  
        im = ax1.imshow(plot_om_search, cmap='hsv')
        ax1.set_title("Mappa delle orientazioni (ricerca)")
        ax2 = fig.add_subplot(gs[2])  
        ax2.imshow(plot_om_match, cmap='hsv')
        ax2.set_title("Mappa delle orientazioni (match)")
        ax3 = fig.add_subplot(gs[3])  
        ax3.imshow(plot_minutiae_search)
        ax3.set_title("Minutiae (ricerca)")
        ax4 = fig.add_subplot(gs[4]) 
        ax4.imshow(plot_minutiae_match)
        ax4.set_title("Minutiae (match)")
        fig.colorbar(im, cax=ax0, label="Orientazione (gradi)")
        for a in [ax1, ax2, ax3, ax4]:
            a.axis("off")
        plt.tight_layout()
        self.match_img=fig

        
def plot_minutiae(img,minutiae): #plot with minutiae points
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for _, row in minutiae.iterrows():
        x, y, m_type = row["x"], row["y"], row["tipo"]
        if m_type == "end":
            cv2.circle(img_color, (x, y), 3, (255, 0, 0), 1)  
        elif m_type == "bifurcation":
            cv2.circle(img_color, (x, y), 2, (0, 0, 255), -1)  
    return img_color

def compute_save_dir(dirpath): #Upload a folder of files
    dirpath=Path(dirpath)
    count=0
    if not dirpath.is_dir():
        raise ValueError(f"{dirpath} non è una directory valida.")
    for img in dirpath.iterdir():
        if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}: 
            compute_save(img)
            count+=1
    return count

def compute_save(filepath): #upload a single file
    filepath=Path(filepath)
    if filepath.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}: 
        raise ValueError('il file non è un immagine')
    try:
        fp=fingerprint(filepath)
        fp.lay()
        s.upload(fp)
    except IOError:
        print(f'Errore durante la lettura del file.{filepath}')
  
def rotate(map,degrees): #rotate stuff
    (h, w) = map.shape
    center = (w // 2, h // 2)
    matrice_rotazione = cv2.getRotationMatrix2D(center, float(degrees), 1.0)
    rotated= cv2.warpAffine(map, matrice_rotazione, (w, h))
    return rotated

def save(fp, find): #save the search
    p=s.save_search(fp, find)
    return p

def check_db(): # checks if there are files in the db
    return s.check_db()
