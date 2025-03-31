import os
import re
import time
import requests

# Fonction pour nettoyer les noms de fichiers et dossiers
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Détermine la tranche d'années en fonction d'une année donnée
def get_year_range(year):
    try:
        year = int(year)
    except (ValueError, TypeError):
        return "Unknown"
    if 2000 <= year <= 2009:
        return "2000-2009"
    elif 2010 <= year <= 2019:
        return "2010-2019"
    elif 2020 <= year <= 2025:
        return "2020-2025"
    else:
        return "Autres"

# Recherche d’un album via l’API MusicBrainz
def search_album_mb(artist, album):
    # Construction de la requête de recherche
    query = f'release:"{album}" AND artist:"{artist}"'
    url = "http://musicbrainz.org/ws/2/release/"
    headers = {
        "User-Agent": "LyricsProject/1.0 (contact@example.com)"
    }
    params = {
        "query": query,
        "fmt": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        print("Erreur MusicBrainz (search) :", response.status_code)
        return None
    data = response.json()
    releases = data.get("releases", [])
    if not releases:
        return None
    # On choisit le premier résultat
    return releases[0]

# Récupère la liste des pistes et d'autres infos via l’API MusicBrainz (avec inc=recordings)
def get_album_tracks_mb(mbid):
    url = f"http://musicbrainz.org/ws/2/release/{mbid}"
    headers = {
        "User-Agent": "LyricsProject/1.0 (contact@example.com)"
    }
    params = {
        "inc": "recordings",
        "fmt": "json"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        print("Erreur MusicBrainz (tracks) :", response.status_code)
        return None
    data = response.json()
    tracks = []
    # Les pistes sont regroupées dans la liste "media"
    for medium in data.get("media", []):
        for track in medium.get("tracks", []):
            tracks.append(track.get("title"))
    return tracks

# Récupère les paroles d'un morceau via l’API Lyrics.ovh
def get_track_lyrics(artist, track):
    url = f"https://api.lyrics.ovh/v1/{artist}/{track}"
    response = requests.get(url)
    if response.status_code != 200:
        # Pas de paroles trouvées ou autre erreur
        return None
    data = response.json()
    return data.get("lyrics", None)

# Liste d'albums à traiter (à adapter)
albums = [
    {"genre": "Rap", "artist": "Booba", "album": "Ouest Side"},
    {"genre": "Rap", "artist": "PNL", "album": "Le Monde Chico"},
    {"genre": "Rap", "artist": "Kaaris", "album": "Or Noir"},
    {"genre": "Rap", "artist": "La Fouine", "album": "Capitale du Crime"},
    {"genre": "Rap", "artist": "Orelsan", "album": "Perdu d'avance"},
    {"genre": "Rap", "artist": "SCH", "album": "JVLIVS"},
    {"genre": "Rap", "artist": "Lomepal", "album": "Jeannine"},
    {"genre": "Rap", "artist": "Nekfeu", "album": "Feu"},
    {"genre": "Rap", "artist": "Lacrim", "album": "Force & Honneur"},
    {"genre": "Rap", "artist": "Jul", "album": "Je trouve pas le sommeil"},
    {"genre": "Musique française", "artist": "Zaz", "album": "Zaz"},
    {"genre": "Musique française", "artist": "Indochine", "album": "13"},
    {"genre": "Musique française", "artist": "M. Pokora", "album": "À la poursuite du bonheur"},
    {"genre": "Musique française", "artist": "Amir", "album": "Au cœur de moi"},
    {"genre": "Musique française", "artist": "Vianney", "album": "Idées blanches"},
    {"genre": "Musique française", "artist": "Clara Luciani", "album": "Sainte-Victoire"},
    {"genre": "Musique française", "artist": "Coeur de Pirate", "album": "Blonde"},
    {"genre": "Musique française", "artist": "Christine and the Queens", "album": "Chris"},
    {"genre": "Musique française", "artist": "Julien Doré", "album": "Vous & moi"},
    {"genre": "Musique française", "artist": "Black M", "album": "Éternel insatisfait"},
    {"genre": "Rock français", "artist": "Noir Désir", "album": "Tostaky"},
    {"genre": "Rock français", "artist": "Trust", "album": "Répression"},
    {"genre": "Rock français", "artist": "Téléphone", "album": "Dure Limite"},
    {"genre": "Rock français", "artist": "Indochine", "album": "L'aventurier"},
    {"genre": "Rock français", "artist": "BB Brunes", "album": "Nico Teen Love"},
    {"genre": "Chanson française", "artist": "Édith Piaf", "album": "La Vie en Rose"},
    {"genre": "Chanson française", "artist": "Jacques Brel", "album": "Amsterdam"},
    {"genre": "Chanson française", "artist": "Charles Aznavour", "album": "La Bohème"},
    {"genre": "Chanson française", "artist": "Francis Cabrel", "album": "Samedi soir sur la terre"},
    {"genre": "Chanson française", "artist": "Zazie", "album": "Zen"}
]


# Dossier racine pour l'enregistrement
root_dir = "lyrics_dataset"

for album in albums:
    genre = album["genre"]
    artist = album["artist"]
    album_title = album["album"]
    
    print(f"Recherche de l'album '{album_title}' de {artist}...")
    album_info = search_album_mb(artist, album_title)
    if album_info is None:
        print(f"  -> Album introuvable pour {artist} - {album_title}.")
        continue
    
    mbid = album_info.get("id")
    release_date = album_info.get("date", "")
    release_year = release_date.split("-")[0] if release_date else "0"
    year_range = get_year_range(release_year)
    
    # Création de l'arborescence : genre / tranche d'années / album
    album_folder = os.path.join(root_dir, sanitize_filename(genre), year_range, sanitize_filename(album_title))
    os.makedirs(album_folder, exist_ok=True)
    
    print(f"  -> Album trouvé (MBID: {mbid}, Année: {release_year}). Récupération des titres...")
    track_list = get_album_tracks_mb(mbid)
    if not track_list:
        print("  -> Aucun titre trouvé pour cet album.")
        continue
    print(f"  -> {len(track_list)} titre(s) trouvé(s).")
    
    for track in track_list:
        print(f"    Traitement du titre : {track}")
        lyrics = get_track_lyrics(artist, track)
        if not lyrics:
            print("      -> Paroles non trouvées pour ce titre.")
            continue
        
        # Création du nom de fichier
        file_name = sanitize_filename(track) + ".txt"
        file_path = os.path.join(album_folder, file_name)
        
        # Écriture du fichier avec les métadonnées en en-tête
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Année : {release_year}\n")
            f.write(f"Album : {album_title}\n")
            f.write(f"Titre : {track}\n")
            f.write(f"Artiste : {artist}\n")
            f.write("\n")
            f.write(lyrics)
        
        print(f"      -> Fichier enregistré : {file_path}")
        time.sleep(0.01)  # Pause entre les titres
    time.sleep(0.1)  # Pause entre les albums
