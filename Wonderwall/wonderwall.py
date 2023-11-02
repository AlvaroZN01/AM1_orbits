import os

directorio = os.getcwd()
ruta = os.path.join(directorio, "Wonderwall/Wonderwall_lyrics_and_chords.txt")

with open(ruta, "r") as archivo:
    lyrics_and_chords_string = archivo.read()

lyrics_and_chords = lyrics_and_chords_string.split("\n")

# Hasta aquí está bien

divisores = ("[Verse 1]", "[Verse 2]", "[Pre-chorus]", "[Chorus]", "[Verse 3]",  "[Pre-chorus]", "[Chorus]")
lyrics = []
chords = []

# Rehacer a mano a partir de aquí

for i in range(len(lyrics_and_chords)):
    if lyrics_and_chords[i] in divisores:
        indice_divisor = i
        break

contador = 0
for i in range(indice_divisor+1, len(lyrics_and_chords)):
    if len(lyrics_and_chords[i]) > 2:
        if contador % 2 == 1:
            chords.append(lyrics_and_chords[i])
        else: 
            lyrics.append(lyrics_and_chords[i])
        contador += 1
    if len(lyrics_and_chords[i]) == 0 or lyrics_and_chords[i] == "\n":
        for j in range(i,len(lyrics_and_chords)):
            if lyrics_and_chords[j] in divisores:
                indice_divisor = j
                break
        contador = 0

#print(chords)
#print(lyrics)