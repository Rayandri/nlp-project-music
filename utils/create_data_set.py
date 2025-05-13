#!/usr/bin/env python3
"""
Dataset creation and collection utilities for lyrics project.
"""
import os
import re
import time
import json
import base64
import logging
import urllib.parse
import requests
from collections import defaultdict

# ---------- config ----------
ROOT_DIR = "lyrics_dataset"
REPORT = "collection_report.json"
LOG_FILE = "collection.log"
SP_LIMIT = 10            # albums par artiste (max)
SLEEP = 0.015         # pour éviter le rate-limit
CID = "" # remplacez par votre Client ID 
SECRET = "" # remplacez par votre Secret 
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(
        LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)


def sanitize(name):
    """Remove forbidden characters from file/folder names."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


def year_range(y):
    """Return a string representing the year range for a given year."""
    try:
        y = int(str(y)[:4])
    except:
        return "Unknown"
    if 1900 <= y <= 1999:
        return "1900-1999"
    if 2000 <= y <= 2009:
        return "2000-2009"
    if 2010 <= y <= 2019:
        return "2010-2019"
    if 2020 <= y <= 2025:
        return "2020-2025"
    if y >= 2026:
        return "2026-2100"
    return "Autre"

# ---------- Spotify helpers ----------


def sp_token():
    """Get a Spotify API token."""
    cid = CID
    secret = SECRET
    if not cid or not secret:
        return None
    hdr = {"Authorization": "Basic " +
           base64.b64encode(f"{cid}:{secret}".encode()).decode()}
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.post("https://accounts.spotify.com/api/token",
                          data={"grant_type": "client_credentials"}, headers=hdr, timeout=10)
            r.raise_for_status()  # Raise exception for HTTP errors
            return r.json().get("access_token")
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Erreur lors de la récupération du token Spotify (tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error("Échec de la récupération du token après plusieurs tentatives")
                return None


def sp_get(url, token):
    """Make a GET request to the Spotify API with retries."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
            r.raise_for_status()  # Raise exception for HTTP errors
            return r.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Erreur lors de la requête à l'API Spotify (tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error(f"Échec de la requête après plusieurs tentatives: {url}")
                return {}


def sp_artist_id(token, name):
    """Get Spotify artist ID from artist name."""
    try:
        q = urllib.parse.quote(name)
        js = sp_get(
            f"https://api.spotify.com/v1/search?type=artist&limit=1&q={q}", token)
        items = js.get("artists", {}).get("items", [])
        return items[0]["id"] if items else None
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Erreur lors de la récupération de l'ID de l'artiste {name}: {e}")
        return None


def sp_top_albums(token, artist_id):
    """Get top albums for a Spotify artist."""
    albums, seen = [], set()
    try:
        js = sp_get(
            f"https://api.spotify.com/v1/artists/{artist_id}/albums?include_groups=album&limit=50&market=FR", token)
        
        for a in js.get("items", []):
            try:
                name, aid = a["name"], a["id"]
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                
                detail = sp_get(f"https://api.spotify.com/v1/albums/{aid}", token)
                
                albums.append({
                    "id": aid,
                    "title": name,
                    "pop": detail.get("popularity", 0),
                    "year": (detail.get("release_date") or "0")[:4],
                    "tracks": [t["name"] for t in detail.get("tracks", {}).get("items", [])]
                })
                
                time.sleep(SLEEP)
            except (KeyError, IndexError, TypeError) as e:
                logging.error(f"Erreur lors du traitement de l'album {a.get('name', 'inconnu')}: {e}")
                continue
            
        albums.sort(key=lambda x: x["pop"], reverse=True)
        return albums[:SP_LIMIT]
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des albums pour l'artiste {artist_id}: {e}")
        return []


# ---------- MusicBrainz fallback ----------
MB_HEADERS = {"User-Agent": "LyricsDataset/2.0 (contact@example.com)"}


def mb_artist_id(name):
    """Get MusicBrainz artist ID from artist name."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.get("https://musicbrainz.org/ws/2/artist", 
                           params={"query": f'artist:"{name}"', "fmt": "json", "limit": 1}, 
                           headers=MB_HEADERS, timeout=10)
            r.raise_for_status()
            js = r.json()
            items = js.get("artists", [])
            return items[0]["id"] if items else None
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            logging.error(f"Erreur lors de la récupération de l'ID MusicBrainz pour {name} (tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error(f"Échec de la récupération de l'ID MusicBrainz pour {name} après plusieurs tentatives")
                return None


def mb_top_albums(artist_id):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.get("https://musicbrainz.org/ws/2/release-group",
                            params={"artist": artist_id, "type": "album",
                                    "status": "official", "fmt": "json", "limit": 100},
                            headers=MB_HEADERS, timeout=10)
            r.raise_for_status()
            js = r.json()
            rgs = js.get("release-groups", [])
            rgs.sort(key=lambda x: x.get("first-release-date", ""), reverse=True)
            
            albums = []
            for rg in rgs[:SP_LIMIT]:
                try:
                    albums.append({
                        "id": rg["id"],
                        "title": rg["title"],
                        "pop": 0,
                        "year": (rg.get("first-release-date") or "0")[:4],
                        "tracks": mb_tracks(rg["id"])
                    })
                except (KeyError, Exception) as e:
                    logging.error(f"Erreur lors du traitement du release group {rg.get('title', 'inconnu')}: {e}")
                    continue
                
            return albums
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Erreur lors de la récupération des albums MusicBrainz pour {artist_id} (tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error(f"Échec de la récupération des albums MusicBrainz pour {artist_id} après plusieurs tentatives")
                return []


def mb_tracks(rgid):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.get(f"https://musicbrainz.org/ws/2/release-group/{rgid}",
                           params={"inc": "releases", "fmt": "json"},
                           headers=MB_HEADERS, timeout=10)
            r.raise_for_status()
            js = r.json()
            rels = js.get("releases", [])
            
            if not rels:
                return []
                
            rid = rels[0]["id"]
            
            r_det = requests.get(f"https://musicbrainz.org/ws/2/release/{rid}",
                              params={"inc": "recordings", "fmt": "json"},
                              headers=MB_HEADERS, timeout=10)
            r_det.raise_for_status()
            det = r_det.json()
            
            tracks = []
            for m in det.get("media", []):
                try:
                    tracks.extend([t["title"] for t in m.get("tracks", [])])
                except (KeyError, TypeError) as e:
                    logging.warning(f"Erreur lors de l'extraction des titres de pistes: {e}")
                    continue
            
            return tracks
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            logging.error(f"Erreur lors de la récupération des pistes pour {rgid} (tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error(f"Échec de la récupération des pistes pour {rgid} après plusieurs tentatives")
                return []

# ---------- lyrics ----------


def lyrics(artist, title):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.get(
                f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}",
                timeout=10)
            
            # If the API returns a 404, the lyrics don't exist
            if r.status_code == 404:
                return None
                
            # For other error status codes, retry
            r.raise_for_status()
            
            # Check if we have valid JSON and "lyrics" in the response
            data = r.json()
            if "lyrics" not in data:
                logging.warning(f"Aucune parole trouvée pour {artist} - {title}")
                return None
                
            return data["lyrics"]
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logging.error(f"Erreur lors de la récupération des paroles pour {artist} - {title} "
                         f"(tentative {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                logging.info("Attente de 60 secondes avant nouvelle tentative...")
                time.sleep(60)  # Wait 1 minute before retrying
            else:
                logging.error(f"Échec de la récupération des paroles pour {artist} - {title} "
                             f"après plusieurs tentatives")
                return None


# ---------- collecte ----------
ARTISTS = [
    # ---------- Rap ----------
    ("Rap", "Booba"), ("Rap", "PNL"), ("Rap", "Orelsan"),
    ("Rap", "Damso"), ("Rap", "Ninho"), ("Rap", "SCH"),
    ("Rap", "Jul"), ("Rap", "Naza"), ("Rap", "Naps"), ("Rap", "Hamza"),
    ("Rap", "Kaaris"), ("Rap", "Kalash"), ("Rap", "Alonzo"),
    ("Rap", "Soolking"), ("Rap", "Lomepal"), ("Rap", "Nekfeu"),
    ("Rap", "Vald"), ("Rap", "Lefa"), ("Rap", "Dinos"),
    ("Rap", "Rohff"), ("Rap", "Sefyu"), ("Rap", "Sniper"),
    ("Rap", "Heuss L'enfoiré"), ("Rap", "Hatik"), ("Rap", "PLK"),
    ("Rap", "ZKR"), ("Rap", "Kalash Criminel"), ("Rap", "Gazo"),
    ("Rap", "Tiakola"), ("Rap", "SDM"), ("Rap", "Koba LaD"),
    ("Rap", "RK"), ("Rap", "Kery James"), ("Rap", "Fianso"),
    ("Rap", "Sexion d'Assaut"), ("Rap", "IAM"), ("Rap", "NTM"),
    ("Rap", "Mister You"), ("Rap", "Youssoupha"), ("Rap", "Disiz"),
    ("Rap", "Lino"), ("Rap", "Oxmo Puccino"), ("Rap", "Chilla"),
    ("Rap", "Alkpote"), ("Rap", "La Fouine"), ("Rap", "Timal"),
    ("Rap", "Zed"), ("Rap", "Kikesa"), ("Rap", "Hös Copperfield"),

    # ---------- Pop française ----------
    ("Pop française", "Angèle"), ("Pop française", "Aya Nakamura"),
    ("Pop française", "Clara Luciani"), ("Pop française", "Louane"),
    ("Pop française", "Julien Doré"), ("Pop française", "Vianney"),

    # ---------- Rock français ----------
    ("Rock français", "Noir Désir"), ("Rock français", "Indochine"),
    ("Rock français", "Téléphone"), ("Rock français", "Matmatah"),
    ("Rock français", "Shaka Ponk"), ("Rock français", "Saez"),

    # ---------- Metal français ----------
    ("Metal", "Gojira"), ("Metal", "Mass Hysteria"),
    ("Metal", "Dagoba"), ("Metal", "AqME"), ("Metal", "Eths"),

    # ---------- R&B / Soul ----------
    ("R&B", "Tayc"), ("R&B", "Dadju"),
    ("R&B", "Shy'm"), ("R&B", "Amel Bent"), ("R&B", "Slimane"),

    # ---------- Electro ----------
    ("Electro", "Daft Punk"), ("Electro", "David Guetta"),
    ("Electro", "Martin Solveig"), ("Electro", "Vitalic"),
    ("Electro", "M83"),

    # ---------- Reggae ----------
    ("Reggae", "Tryo"), ("Reggae", "Danakil"),
    ("Reggae", "Naâman"), ("Reggae", "Pierpoljak"),
    ("Reggae", "Dub Inc"),

    # ---------- Chanson française ----------
    ("Chanson française", "Serge Gainsbourg"), ("Chanson française", "Jacques Brel"),
    ("Chanson française", "Charles Aznavour"), ("Chanson française", "Francis Cabrel"),
    ("Chanson française", "Renaud"), ("Chanson française", "Zaz"),
]

token = sp_token()
report = defaultdict(
    lambda: {"status": "incomplet", "albums": [], "missing": []})

for genre, artist in ARTISTS:
    logging.info(f"=== {artist} ({genre}) ===")
    albs = []
    if token:
        aid = sp_artist_id(token, artist)
        if aid:
            albs = sp_top_albums(token, aid)
    if len(albs) < SP_LIMIT:
        logging.warning("Spotify incomplet, fallback MusicBrainz")
        mbid = mb_artist_id(artist)
        if mbid:
            albs = mb_top_albums(mbid)
    if len(albs) < SP_LIMIT:
        report[artist]["missing"] = [a["title"] for a in albs]
        logging.error("Moins de 5 albums, artiste ignoré")
        continue

    for alb in albs:
        yr = year_range(alb["year"])
        try:
            folder = os.path.join(ROOT_DIR, yr, sanitize(
                genre), f"{sanitize(alb['title'])}-artist-{sanitize(artist)}")
            os.makedirs(folder, exist_ok=True)
            
            tracks_with_lyrics = 0
            for tr in alb["tracks"]:
                try:
                    lyr = lyrics(artist, tr)
                    if not lyr:
                        logging.warning(f"Pas de paroles pour {artist} - {tr}")
                        continue
                        
                    file_path = os.path.join(folder, sanitize(tr) + ".txt")
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(lyr)
                        tracks_with_lyrics += 1
                    except IOError as e:
                        logging.error(f"Erreur lors de l'écriture du fichier {file_path}: {e}")
                        # Wait and try again if it's a temporary issue
                        time.sleep(60)
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(lyr)
                            tracks_with_lyrics += 1
                            logging.info(f"Réussite de l'écriture du fichier {file_path} après attente")
                        except IOError as e2:
                            logging.error(f"Échec persistant d'écriture du fichier {file_path}: {e2}")
                except Exception as e:
                    logging.error(f"Erreur lors du traitement de la piste {tr} pour {artist}: {e}")
                    continue
                
                time.sleep(SLEEP)
                
            if tracks_with_lyrics > 0:
                report[artist]["albums"].append(alb["title"])
                logging.info(f"Album {alb['title']} : {tracks_with_lyrics} pistes avec paroles")
            else:
                logging.warning(f"Album {alb['title']} : aucune piste avec paroles")
                
        except Exception as e:
            logging.error(f"Erreur lors du traitement de l'album {alb.get('title', 'inconnu')} pour {artist}: {e}")
            continue
            
    if report[artist]["albums"]:
        report[artist]["status"] = "complet"
        logging.info(f"Artiste {artist} complété avec {len(report[artist]['albums'])} albums")
    else:
        logging.warning(f"Aucun album avec paroles pour {artist}")
    logging.info("OK")

try:
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Rapport de collecte écrit dans {REPORT}")
except IOError as e:
    logging.error(f"Erreur lors de l'écriture du rapport: {e}")
    # Try again after waiting
    time.sleep(60)
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logging.info(f"Rapport de collecte écrit dans {REPORT} après attente")
    except IOError as e2:
        logging.error(f"Échec persistant d'écriture du rapport: {e2}")

logging.info("Collecte terminée")
