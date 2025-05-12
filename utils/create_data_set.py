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
SP_LIMIT = 5            # albums par artiste (mini)
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
    r = requests.post("https://accounts.spotify.com/api/token",
                      data={"grant_type": "client_credentials"}, headers=hdr, timeout=10)
    return r.json().get("access_token")


def sp_get(url, token):
    return requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10).json()


def sp_artist_id(token, name):
    q = urllib.parse.quote(name)
    js = sp_get(
        f"https://api.spotify.com/v1/search?type=artist&limit=1&q={q}", token)
    items = js.get("artists", {}).get("items", [])
    return items[0]["id"] if items else None


def sp_top_albums(token, artist_id):
    albums, seen = [], set()
    js = sp_get(
        f"https://api.spotify.com/v1/artists/{artist_id}/albums?include_groups=album&limit=50&market=FR", token)
    for a in js.get("items", []):
        name, aid = a["name"], a["id"]
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        detail = sp_get(f"https://api.spotify.com/v1/albums/{aid}", token)
        albums.append({"id": aid,
                       "title": name,
                       "pop": detail.get("popularity", 0),
                       "year": (detail.get("release_date") or "0")[:4],
                       "tracks": [t["name"] for t in detail.get("tracks", {}).get("items", [])]})
        time.sleep(SLEEP)
    albums.sort(key=lambda x: x["pop"], reverse=True)
    return albums[:SP_LIMIT]


# ---------- MusicBrainz fallback ----------
MB_HEADERS = {"User-Agent": "LyricsDataset/2.0 (contact@example.com)"}


def mb_artist_id(name):
    js = requests.get("https://musicbrainz.org/ws/2/artist", params={
                      "query": f'artist:"{name}"', "fmt": "json", "limit": 1}, headers=MB_HEADERS, timeout=10).json()
    items = js.get("artists", [])
    return items[0]["id"] if items else None


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

# ---------- lyrics ----------


def lyrics(artist, title):
    r = requests.get(
        f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}", timeout=10)
    if r.status_code != 200:
        return None
    return r.json().get("lyrics")


# ---------- collecte ----------
ARTISTS = [
    ("Rap", "Booba"), ("Rap", "PNL"), ("Rap", "Kaaris"), ("Rap", "La Fouine"),
    ("Rap", "Orelsan"), ("Rap", "SCH"), ("Rap", "Lomepal"), ("Rap", "Nekfeu"),
    ("Rap", "Lacrim"), ("Rap", "Jul"),
    ("Musique française", "Zaz"), ("Musique française", "Indochine"),
    ("Musique française", "M. Pokora"), ("Musique française", "Amir"),
    ("Musique française", "Vianney"),
    ("Rock français", "Noir Désir"), ("Rock français", "Trust"),
    ("Rock français", "Téléphone"), ("Rock français", "BB Brunes"),
    ("Chanson française", "Édith Piaf"), ("Chanson française", "Jacques Brel"),
    ("Chanson française", "Charles Aznavour"), ("Chanson française", "Francis Cabrel"),
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
