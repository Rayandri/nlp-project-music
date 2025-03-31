import os
import re
import time
import requests
from bs4 import BeautifulSoup
import lyricsgenius

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

# Construit l'URL d'un album sur Genius
# Remarque : la structure attendue est "https://genius.com/albums/Artiste/Album"
def build_album_url(artist, album):
    artist_url = sanitize_filename(artist).replace(" ", "-")
    album_url = sanitize_filename(album).replace(" ", "-")
    return f"https://genius.com/albums/{artist_url}/{album_url}"

# Récupère la liste des URL des chansons depuis la page d'un album Genius
def get_album_songs(album_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0)"
    }
    try:
        response = requests.get(album_url, headers=headers)
    except Exception as e:
        print("Erreur de connexion pour", album_url, e)
        return []
    if response.status_code != 200:
        print("Erreur lors de la récupération de l'album :", album_url)
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    song_links = []
    # Les pages d'album sur Genius affichent généralement les chansons sous forme de "mini cards"
    for a in soup.find_all('a', class_='mini_card'):
        href = a.get('href')
        if href and '/lyrics' in href:
            song_links.append(href)
    return song_links

# Extrait l'année de sortie de l'album en scrappant la page (recherche dans les blocs de métadonnées)
def get_album_year(album_url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0)"}
    try:
        response = requests.get(album_url, headers=headers)
    except Exception as e:
        print("Erreur de connexion pour", album_url, e)
        return None
    if response.status_code != 200:
        print("Erreur lors de la récupération de l'album pour l'année :", album_url)
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    # Cherche dans les blocs de métadonnées (la classe peut varier)
    metadata_units = soup.find_all("div", class_="metadata_unit-info")
    for unit in metadata_units:
        text = unit.get_text(strip=True)
        # On cherche un motif d'année (par ex. 2006, 2010, etc.)
        match = re.search(r"(20\d{2})", text)
        if match:
            return int(match.group(1))
    return None

# Récupère les paroles d'une chanson à partir de son URL
def get_lyrics_from_url(song_url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0)"}
    try:
        response = requests.get(song_url, headers=headers)
    except Exception as e:
        print("Erreur de connexion pour", song_url, e)
        return None
    if response.status_code != 200:
        print("Erreur lors de la récupération de la chanson :", song_url)
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    # Tentative avec la nouvelle structure de Genius
    lyrics_div = soup.find("div", class_="Lyrics__Root")
    if not lyrics_div:
        # Fallback à l'ancienne structure
        lyrics_div = soup.find("div", class_="lyrics")
    if lyrics_div:
        return lyrics_div.get_text(separator="\n").strip()
    return None

# (Optionnel) Récupère le titre de la chanson depuis la page
def get_song_title(song_url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DataScraper/1.0)"}
    try:
        response = requests.get(song_url, headers=headers)
    except Exception as e:
        print("Erreur de connexion pour", song_url, e)
        return "Unknown Title"
    if response.status_code != 200:
        return "Unknown Title"
    soup = BeautifulSoup(response.text, 'html.parser')
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text()
        # Généralement le titre se présente sous la forme "Titre Lyrics | Artiste | Genius"
        title = title_text.split("Lyrics")[0].strip()
        return title
    return "Unknown Title"

# Liste d'albums à traiter.
# Pour chaque album, on fournit le genre, l'artiste et le nom de l'album.
# Vous pouvez enrichir ou modifier cette liste selon vos besoins.
albums = [
    {"genre": "Rap", "artist": "Booba", "album": "Ouest Side"},
    {"genre": "Rap", "artist": "PNL", "album": "Le Monde Chico"},
    {"genre": "Pop", "artist": "Lady Gaga", "album": "The Fame"},
    {"genre": "Pop", "artist": "Dua Lipa", "album": "Future Nostalgia"},
    {"genre": "Musique française", "artist": "Zaz", "album": "Zaz"},
    {"genre": "Musique française", "artist": "Indochine", "album": "13"}
]

# Dossier racine où seront enregistrés les fichiers
root_dir = "lyrics_dataset"

# (Optionnel) Initialisation de l'API Genius pour d'éventuelles recherches complémentaires
genius = lyricsgenius.Genius("cl2NHO8b_z_1cbU3-VOOxkGsV7892ILTIEuAjLTkXtJc8AgB3rsPMKze86tpakr1", timeout=15, retries=3)

# Traitement de chaque album
for album in albums:
    genre = album["genre"]
    artist = album["artist"]
    album_title = album["album"]
    
    album_url = build_album_url(artist, album_title)
    print(f"Traitement de l'album '{album_title}' de {artist} : {album_url}")
    
    # Récupère l'année de sortie via le scraping de la page album
    release_year = get_album_year(album_url)
    if release_year is None:
        release_year = 0  # ou "Unknown"
    year_range = get_year_range(release_year)
    
    # Création de la structure de dossiers : root/genre/year_range/album
    album_folder = os.path.join(root_dir, sanitize_filename(genre), year_range, sanitize_filename(album_title))
    os.makedirs(album_folder, exist_ok=True)
    
    # Récupération de la liste des chansons dans l'album
    song_urls = get_album_songs(album_url)
    print(f"  {len(song_urls)} chanson(s) trouvée(s).")
    
    # Pour chaque chanson, récupère le titre et les paroles, puis enregistre dans un fichier
    for song_url in song_urls:
        print(f"  Traitement de la chanson : {song_url}")
        title = get_song_title(song_url)
        lyrics = get_lyrics_from_url(song_url)
        if not lyrics:
            print("    Paroles non trouvées pour cette chanson.")
            continue
        
        # Création du nom du fichier
        file_name = sanitize_filename(title) + ".txt"
        file_path = os.path.join(album_folder, file_name)
        
        # Écriture du fichier texte avec les métadonnées en en-tête
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Année : {release_year}\n")
            f.write(f"Album : {album_title}\n")
            f.write(f"Titre : {title}\n")
            f.write(f"Artiste : {artist}\n")
            f.write("\n")
            f.write(lyrics)
        
        print(f"    -> Fichier enregistré : {file_path}")
        time.sleep(1)  # pause entre les chansons
    time.sleep(2)  # pause entre les albums
