#!/usr/bin/env python3
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
SP_LIMIT = 3            # albums par artiste (mini)
SLEEP = 0.015         # pour éviter le rate-limit
CID = "4dab290d280d4d06af8d029195c90e2c"
SECRET = "bf7127954d454b17af2ac620f83e7153"
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(
        LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)


def sanitize(name): return re.sub(r'[\\/*?:"<>|]', "", name)


def year_range(y):
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
    js = requests.get("https://musicbrainz.org/ws/2/release-group",
                      params={"artist": artist_id, "type": "album",
                              "status": "official", "fmt": "json", "limit": 100},
                      headers=MB_HEADERS, timeout=10).json()
    rgs = js.get("release-groups", [])
    rgs.sort(key=lambda x: x.get("first-release-date", ""), reverse=True)
    albums = []
    for rg in rgs[:SP_LIMIT]:
        albums.append({"id": rg["id"],
                       "title": rg["title"],
                       "pop": 0,
                       "year": (rg.get("first-release-date") or "0")[:4],
                       "tracks": mb_tracks(rg["id"])})
    return albums


def mb_tracks(rgid):
    js = requests.get(f"https://musicbrainz.org/ws/2/release-group/{rgid}",
                      params={"inc": "releases", "fmt": "json"},
                      headers=MB_HEADERS, timeout=10).json()
    rels = js.get("releases", [])
    if not rels:
        return []
    rid = rels[0]["id"]
    det = requests.get(f"https://musicbrainz.org/ws/2/release/{rid}",
                       params={"inc": "recordings", "fmt": "json"},
                       headers=MB_HEADERS, timeout=10).json()
    tracks = []
    for m in det.get("media", []):
        tracks.extend([t["title"] for t in m.get("tracks", [])])
    return tracks
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
        folder = os.path.join(ROOT_DIR, yr, sanitize(
            genre), f"{sanitize(alb['title'])}-artist-{sanitize(artist)}")
        os.makedirs(folder, exist_ok=True)
        for tr in alb["tracks"]:
            lyr = lyrics(artist, tr)
            if not lyr:
                continue
            with open(os.path.join(folder, sanitize(tr) + ".txt"), "w", encoding="utf-8") as f:
                f.write(lyr)
            time.sleep(SLEEP)
        report[artist]["albums"].append(alb["title"])
    report[artist]["status"] = "complet"
    logging.info("OK")

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

logging.info("Collecte terminée")
