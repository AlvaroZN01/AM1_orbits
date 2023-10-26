import os

directorio = os.getcwd()
ruta = os.path.join(directorio, "Wonderwall/Wonderwall_lyrics_and_chord.txt")

with open(ruta, "r") as archivo:
    lyrics_and_chords_string = archivo.read()

lyrics_and_chords = lyrics_and_chords_string.split("\n")

lyrics = []
for linea in lyrics_and_chords:
    if not linea.startswith("["):
        lyrics.append(linea)

# Buscar [Verse 1], [Verse 2], [Pre-chorus]... y coger pares e impares hasta que se encuentre un salto de línea. 
# Entonces pasar al siguiente. 

print(lyrics)