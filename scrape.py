import os
import re
import time
import requests

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

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
    elif year < 2000 and year > 1900:
        return "1900-1999"
    elif year > 2025:
        return "2026-2100"
    return "Autre"

def search_album_mb(artist, album):
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
    return releases[0]

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
    for medium in data.get("media", []):
        for track in medium.get("tracks", []):
            tracks.append(track.get("title"))
    return tracks

def get_track_lyrics(artist, track):
    url = f"https://api.lyrics.ovh/v1/{artist}/{track}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("lyrics", None)

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
    {"genre": "Chanson française", "artist": "Zazie", "album": "Zen"},
    {"genre": "Rap", "artist": "IAM", "album": "L’école du micro d’argent"},
    {"genre": "Rap", "artist": "NTM", "album": "Suprême NTM"},
    {"genre": "Rap", "artist": "Soprano", "album": "Cosmopolitanie"},
    {"genre": "Rap", "artist": "Vald", "album": "Agartha"},
    {"genre": "Rap", "artist": "Niska", "album": "Commando"},
    {"genre": "Rap", "artist": "Ninho", "album": "Destin"},
    {"genre": "Rap", "artist": "Rohff", "album": "La fierté des nôtres"},
    {"genre": "Rap", "artist": "Lefa", "album": "Fame"},
    {"genre": "Rap", "artist": "Alkpote", "album": "Les Marches de l’Empereur"},
    {"genre": "Rap", "artist": "Dinos", "album": "Stamina"},
    {"genre": "Rap", "artist": "PLK", "album": "Enna"},
    {"genre": "Rap", "artist": "ZKR", "album": "Dans les mains"},
    {"genre": "Rap", "artist": "Kalash Criminel", "album": "Sélection naturelle"},
    {"genre": "Rap", "artist": "Kery James", "album": "Mouhammad Alix"},
    {"genre": "Rap", "artist": "Youssoupha", "album": "NGRTD"},
    {"genre": "Rap", "artist": "Disiz", "album": "Pacifique"},
    {"genre": "Rap", "artist": "Mister You", "album": "Dans ma grotte"},
    {"genre": "Rap", "artist": "Heuss L’enfoiré", "album": "En esprit"},
    {"genre": "Rap", "artist": "Hatik", "album": "Vague à l’âme"},
    {"genre": "Rap", "artist": "Chilla", "album": "MŪN"},
    {"genre": "Rap", "artist": "Dua Lipa", "album": "Future Nostalgia"},
    {"genre": "Rap", "artist": "Damso", "album": "Lithopédion"},
    {"genre": "Rap", "artist": "Niro", "album": "Les autres c'est nous"},
    {"genre": "Rap", "artist": "Sofiane", "album": "Affranchis"},
    {"genre": "Rap", "artist": "Kool Shen", "album": "Sur le fil"},
    {"genre": "Rap", "artist": "Sexion d'Assaut", "album": "L'Apogée"},
    {"genre": "Rap", "artist": "Lino", "album": "Psycho-Mélancolie"},
    {"genre": "Rap", "artist": "Oxmo Puccino", "album": "Lipopette Bar"},
    {"genre": "Rap", "artist": "Fianso", "album": "#JeSuisPasséChezSo"},
    {"genre": "Rap", "artist": "Niro", "album": "#CoupDePression"},
    {"genre": "Chanson française", "artist": "Georges Brassens", "album": "Les Copains d’abord"},
    {"genre": "Chanson française", "artist": "Serge Gainsbourg", "album": "Histoire de Melody Nelson"},
    {"genre": "Chanson française", "artist": "Barbara", "album": "L’aigle noir"},
    {"genre": "Chanson française", "artist": "Michel Sardou", "album": "La maladie d’amour"},
    {"genre": "Chanson française", "artist": "Maxime Le Forestier", "album": "Mon frère"},
    {"genre": "Chanson française", "artist": "Alain Bashung", "album": "Osez Joséphine"},
    {"genre": "Chanson française", "artist": "Julien Clerc", "album": "Fou, peut-être"},
    {"genre": "Chanson française", "artist": "Renaud", "album": "Mistral gagnant"},
    {"genre": "Chanson française", "artist": "Léo Ferré", "album": "Amour Anarchie"},
    {"genre": "Chanson française", "artist": "Jean-Jacques Goldman", "album": "Positif"},
    {"genre": "Chanson française", "artist": "Michel Delpech", "album": "Wight Is Wight"},
    {"genre": "Chanson française", "artist": "Patrick Bruel", "album": "Alors regarde"},
    {"genre": "Chanson française", "artist": "Florent Pagny", "album": "Savoir aimer"},
    {"genre": "Chanson française", "artist": "Calogero", "album": "3"},
    {"genre": "Chanson française", "artist": "Michel Berger", "album": "Beauséjour"},
    {"genre": "Rock français", "artist": "Matmatah", "album": "La Ouache"},
    {"genre": "Rock français", "artist": "Saez", "album": "Jours étranges"},
    {"genre": "Rock français", "artist": "Luke", "album": "La tête en arrière"},
    {"genre": "Rock français", "artist": "No One Is Innocent", "album": "Revolution.com"},
    {"genre": "Rock français", "artist": "Dionysos", "album": "Western sous la neige"},
    {"genre": "Rock français", "artist": "Shaka Ponk", "album": "The Geeks and the Jerkin' Socks"},
    {"genre": "Rock français", "artist": "Astonvilla", "album": "Live"},
    {"genre": "Rock français", "artist": "Eiffel", "album": "Le quart d’heure des ahuris"},
    {"genre": "Rock français", "artist": "Kaolin", "album": "De retour dans nos criques"},
    {"genre": "Rock français", "artist": "Zazie", "album": "Rodéo"},
    {"genre": "Musique française", "artist": "Baloji", "album": "137 Avenue Kaniama"},
    {"genre": "Musique française", "artist": "Lara Fabian", "album": "Nue"},
    {"genre": "Musique française", "artist": "Pierre Lapointe", "album": "La science du cœur"},
    {"genre": "Musique française", "artist": "Cœur de pirate", "album": "Roses"},
    {"genre": "Musique française", "artist": "Salvatore Adamo", "album": "Mes chansons italiennes"},
    {"genre": "Musique française", "artist": "Safia Nolin", "album": "Limoilou"},
    {"genre": "Musique française", "artist": "Jean Leloup", "album": "Le Dôme"},
    {"genre": "Musique française", "artist": "Daniel Bélanger", "album": "Rêver mieux"},
    {"genre": "Musique française", "artist": "Dumas", "album": "Nord"},
    {"genre": "Musique française", "artist": "Ariane Moffatt", "album": "22h22"}
]


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
        # print(f"    Traitement du titre : {track}")
        lyrics = get_track_lyrics(artist, track)
        if not lyrics:
            print(f"      -> Paroles non trouvées pour {track}, {artist}.")
            continue
        
        file_name = sanitize_filename(track) + ".txt"
        file_path = os.path.join(album_folder, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(lyrics)
        time.sleep(0.0001)
    time.sleep(0.0001)
