import os
import time
from pathlib import Path
import core



def main_menu(): #just the main menu. Here the user chooses wether to search or upload a fingerprint
    while True:
        clear_screen()
        print( "=" * 30 + " MENU " + "=" * 30)
        print('\nSalve, questo programma permette di analizzare e confrontare impronte digitali. Ecco una breve overview delle funzionalità disponibili ')
        print("1. Hai già caricato il tuo dataset e vuoi cercare una corrispondenza?")
        print("Digita 1 per cercare una corrispondenza")
        print("2. Vuoi fare l'upload di una o più impronte che hai raccolto per analisi future?")
        print("Digita 2 per eseguire l'upload")
        print("q. Esci")
        print("="*100)
        while ((x := input("Seleziona un'opzione: ").strip().lower()) not in {'1','2','q'}):
            print("\nScelta non valida! Riprova.")
        if x == '1':
            path=retrieving_path()
            if path:
                gradi,step=setting_parameters()
                search(path,gradi,step)
        elif x == '2':
            path=retrieving_path_upload()
            if path:
                upload(path)
        elif x == 'q':
            print("\nUscita dal programma...")
            time.sleep(1)
            clear_screen()
            break

def retrieving_path(): #collects the path and makes sure that it exists
    clear_screen()
    if not core.check_db():          # restituisce true se il db è carico
        print("=" * 30 + " ACCESSO NON CONSENTITO " + "=" * 30)
        print('\nAl momento non sono presenti impronte digitali nel database o esso non risponde.')
        print("Se questo è il tuo primo accesso, carica un set di impronte digitali su disco prima di effettuare una ricerca (opzione 2 nel menu)")
        print("="*100)
        input("\nPremi invio per tornare al Menu principale!")
        return
        
             
    print("=" * 30 + " CERCA CORRISPONDENZE NEL DATABASE " + "=" * 30)
    print('\nUn attimo di attenzione, lascia che ti spieghi in 5 punti come avviene la ricerca. ')
    print("1. La ricerca avviene per singole impronte, una per volta.  ")
    print("2. Avrò bisogno del percorso preciso dell'impronta che vuoi cercare.")
    print("3. Formati supportati: .jpg, .jpeg, .png, .gif, .bmp, .tiff")
    print("4. Assicurati che l'immagine sia pulita.")
    print("5. Il confronto utilizza la mappa delle orientazioni e le minutiae. ")
    print("\nDigita q per tornare al Menu")
    print("="*100)
    
    while (
        percorso := input("Digita il percorso dell'impronta che vuoi cercare: ").strip("\"'").strip().lower()) != 'q' and not (
        (p := Path(percorso)).is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}):
        print("Percorso non valido! Riprova.")

    if percorso == 'q':
        print("Uscita dalla ricerca...")
        time.sleep(1)
        clear_screen()
        return False
    
    return Path(percorso).resolve().as_posix()

def setting_parameters(): #allows the user to set parameters for the research
    clear_screen()
    gradi=None 
    step=None
    print("=" * 30 + " SETTING PARAMETRI " + "=" * 30)
    print("Per cercare migliori corrispondenze verrà applicata una rotazione all'impronta. ")
    print("Le impostazioni di default per la rotazione sono: da -15° a +15° con step di 1° ")

    while (y := input("Vuoi modificare i parametri di rotazione? (si/no) ").strip().lower()) not in {'si', 'no'}:
        print("Risposta non valida. Digita 'si' o 'no'.")
    if y == 'si':
        print("\nLa rotazione si svolge da -n a n. Un'alta rotazione massima e uno step basso rallentano l'elaborazione.")
        print("Cerca di tenere un rapporto (rotazione massima):(step) intorno a 15:1.")

        while not (gradi := input("Gradi di rotazione massima (1-180): ").strip()).isdigit() or not (1 <= (gradi := int(gradi)) <= 180):
            print("Valore non valido. Inserisci un numero intero tra 1 e 180.")

        while not (step := input(f"Step (1-{gradi}): ").strip()).isdigit() or not (1 <= (step := int(step)) <= gradi):
            print(f"Valore non valido. Inserisci un numero intero tra 1 e {gradi}.")
    clear_screen()
    print('INPUT RACCOLTI CON SUCCESSO! INIZIO ANALISI... ')
    time.sleep(2)
    return(gradi,step)

def search(path,gradi,step): #performs the research
    clear_screen()
    print("=" * 30 +  f"RICERCA DI {Path(path).stem} IN CORSO"  + "=" * 30)
    print('\nCreazione oggetto impronta...')
    wanted=core.fingerprint(path)
    wanted.lay()
    find=core.search(wanted)
    print('Confronto tramite mappe delle orientazioni in corso...')
    if gradi is None and step is None:
        count=find.find_similar_maps()
    else:
        count=find.find_similar_maps(maxdegrees=gradi, step=step)

    print(f"Confrontate {count} mappe delle orientazioni!\n")
    print('CLASSIFICA PROVVISORIA DEI MIGLIORI MATCH:')
    for nome_file, errore, rotazione in find.top5maps:
            print(f"{nome_file} - Errore: {errore:.4f} - Rotazione: {rotazione} gradi")
    print("\nEstrazione delle minutiae in corso...")
    find.rotate_extract()
    print('Estrazione completata! \nProcedura di matching in corso...')
    find.best_match_minutiae()
    print("Procedura terminata con successo!\n")
    print("=" * 30 + "MIGLIORI MATCH!"  + "=" * 30)
    print("Impronta più simile: ", find.matches[0][0])
    print("Lista punteggi finali:")
    for filename, score in find.matches:
        print(f"{filename} - Punteggio finale: {score:.4f}")
    while (y := input("\nVuoi visualizzare i grafici? (si/no) ").strip().lower()) not in {'si', 'no'}:
        print("Risposta non valida. Digita 'si' o 'no'.")
    if y=='si':
        visualize(wanted, find)
    while (y := input("\nVuoi salvare la ricerca? (si/no) ").strip().lower()) not in {'si', 'no'}:
        print("Risposta non valida. Digita 'si' o 'no'.")
    if y=='si':
        p=core.save(wanted, find)
        print(f'Ricerca salvata con successo! Percorso: {p}')

    input('Premi invio per tornare al menu iniziale!')
    
def visualize(wanted, find): #elaborates the plots
    find.visual_match()
    find.match_img.show()
    wanted.visual_steps()
    wanted.steps_img.show()

def retrieving_path_upload(): #collects the path for the upload and makes sure that it exists
    clear_screen()
    print("=" * 30 + " CARICA NUOVE IMPRONTE NEL DATABASE " + "=" * 30)
    print('\nUn attimo di attenzione, lascia che ti spieghi in breve come avviene il caricamento in 5 punti. ')
    print("1. Puoi caricare una o più impronte digitali insieme.")
    print("2. Avrò bisogno del percorso preciso del file o della cartella contente le più impronte.")
    print("3. Formati supportati: .jpg, .jpeg, .png, .gif, .bmp, .tiff")
    print("4. Assicurati che l'immagine sia pulita.")
    print("5. Di ogni impronta salverò un'immagine ripulita, mappa delle orientazioni in formato .npy e un dataframe contente le minutiae. ")
    print("\nDigita q per tornare al Menu")
    print("="*100)

    while (
        percorso := input("Digita il percorso del file o della cartella che vuoi caricare: ").strip("\"'").strip().lower()) != 'q' and not ((
        (p := Path(percorso)).is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}) or p.is_dir()):
        print("Percorso non valido! Riprova.")

    if percorso == 'q':
        print("Uscita dall'upload...")
        time.sleep(1)
        clear_screen()
        return False

    return(Path(percorso))

def upload(percorso):# performs the upload whether it's one file or one folder  .
    try:
        if percorso.is_dir():
            dim=len(os.listdir(percorso))
            print(f'\nCaricamento di {dim} elementi in corso...')
            count=core.compute_save_dir(percorso)
            print(f"Caricamento di {count} impronte avvenuto con successo su {dim} elementi totali nella cartella {percorso} ")
        elif percorso.is_file():
            print(f'\nCaricamento del file {percorso} in corso...')    
            core.compute_save(percorso)
            print(f'Caricamento avvenuto con successo!')
        
        input('\nPremi invio per tornare al menu principale!')

    except Exception as errore:
        print(f"Si è verificato un errore durante il caricamento: {errore}")

    
def clear_screen(): #clears screen
    os.system("cls" if os.name == "nt" else "clear")
