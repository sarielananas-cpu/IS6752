import os, requests, json, time
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
assert API_KEY, ""The environment variable did not read GOOGLE_MAPS_API_KEY"

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
HEADERS_PROBE = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": ",".join([
        "places.id","places.displayName","places.rating","places.userRatingCount",
        "places.types","places.location","places.priceLevel","places.googleMapsUri"
    ])
}
PAYLOAD_PROBE = {
    "includedTypes": ["cafe"],
    "maxResultCount": 5,
    "locationRestriction": {"circle": {
        "center": {"latitude": 1.3048, "longitude": 103.8318},  # Orchard
        "radius": 1500
    }}
}

r = requests.post(NEARBY_URL, headers=HEADERS_PROBE, json=PAYLOAD_PROBE, timeout=30)
print("status:", r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:1500])

import math
MIN_LAT, MAX_LAT = 1.200, 1.470
MIN_LNG, MAX_LNG = 103.600, 104.100
RADIUS_M = 1800

LAT_M_PER_DEG = 111_320
LNG_M_PER_DEG = math.cos(math.radians(1.35)) * 111_320
lat_step = (RADIUS_M * 1.6) / LAT_M_PER_DEG
lng_step = (RADIUS_M * 1.6) / LNG_M_PER_DEG

def gen_grid(min_lat,max_lat,min_lng,max_lng,lat_step,lng_step):
    lats, lngs = [], []
    cur = min_lat
    while cur <= max_lat: lats.append(round(cur,6)); cur += lat_step
    cur = min_lng
    while cur <= max_lng: lngs.append(round(cur,6)); cur += lng_step
    centers = []
    for i, la in enumerate(lats):
        off = 0.0 if i%2==0 else lng_step/2
        for lo in (x+off for x in lngs):
            if lo <= max_lng: centers.append((la, round(lo,6)))
    return centers

centers = gen_grid(MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG, lat_step, lng_step)
len(centers)
import time, requests
HEADERS = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": ",".join([
        "places.id","places.displayName","places.rating","places.userRatingCount",
        "places.types","places.location","places.priceLevel","places.googleMapsUri"
    ])
}

def search_nearby_cafes(lat, lng, radius_m=RADIUS_M, max_pages=3, verbose=False):
    payload = {
        "includedTypes": ["cafe"],
        "maxResultCount": 20,
        "locationRestriction": {"circle": {
            "center": {"latitude": lat, "longitude": lng},
            "radius": radius_m
        }}
    }
    results, page_token = [], None
    for _ in range(max_pages):
        body = dict(payload)
        if page_token:
            body["pageToken"] = page_token
            time.sleep(1.2)   # 让 token 生效
        r = requests.post(NEARBY_URL, headers=HEADERS, json=body, timeout=30)
        if r.status_code != 200:
            if verbose: print("HTTP", r.status_code, "->", r.text[:500])
            r.raise_for_status()
        j = r.json()
        results += j.get("places", [])
        page_token = j.get("nextPageToken")
        if not page_token: break
    return results
from tqdm import tqdm
all_places, errors, shown = {}, 0, False

for (la, lo) in tqdm(centers, desc="Fetching grid centers"):
    try:
        places = search_nearby_cafes(la, lo, max_pages=3, verbose=not shown)
        for p in places:
            pid = p.get("id")
            if not pid or pid in all_places: continue
            all_places[pid] = {
                "place_id": pid,
                "name": (p.get("displayName") or {}).get("text"),
                "rating": p.get("rating"),
                "userRatingCount": p.get("userRatingCount"),
                "types": ",".join(p.get("types", [])),
                "lat": (p.get("location") or {}).get("latitude"),
                "lng": (p.get("location") or {}).get("longitude"),
                "priceLevel": p.get("priceLevel"),
                "maps_url": p.get("googleMapsUri"),
            }
    except Exception as e:
        errors += 1
        if not shown:
            print("Encountered error at", (la, lo), "->", repr(e))
            shown = True
        continue

len(all_places), errors
import pandas as pd
df = pd.DataFrame.from_dict(all_places, orient="index").reset_index(drop=True)
df.to_csv("sg_cafes_places_v1.csv", index=False, encoding="utf-8")
("sg_cafes_places_v1.csv", df.shape)

import pandas as pd
import numpy as np

csv_path = "sg_cafes_places_v1.csv"  # If not in the current directory, use the full path
df = pd.read_csv(csv_path)

# Basic data cleaning
df['userRatingCount'] = pd.to_numeric(df['userRatingCount'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

print("Total rows:", len(df))
print("Entries with ratings:", df['rating'].notna().sum())
print("Average rating (all):", df['rating'].mean())
print("Median review count (all):", df['userRatingCount'].median())

# Simple duplicate check (place_id should be unique)
dup_ids = df['place_id'].duplicated().sum()
print("Number of duplicate place_ids:", dup_ids)
# At least 20 comments are more robust
df20 = df[df['userRatingCount'].fillna(0) >= 20].copy()

top25 = df20.sort_values(['rating','userRatingCount'], ascending=[False, False])[
    ['name','rating','userRatingCount','priceLevel','maps_url']
].head(25).reset_index(drop=True)

top25
import matplotlib.pyplot as plt

# Rating Distribution (All)
plt.figure()
df['rating'].dropna().plot(kind='hist', bins=20, title='Rating distribution (All cafes)')
plt.xlabel('rating'); plt.ylabel('count'); plt.show()

# Rating Distribution (≥20 comments)
plt.figure()
df20['rating'].dropna().plot(kind='hist', bins=20, title='Rating distribution (>=20 reviews)')
plt.xlabel('rating'); plt.ylabel('count'); plt.show()
# Rating vs. Number of Comments (≥20 comments)
plt.figure()
df20.plot(kind='scatter', x='userRatingCount', y='rating', title='Rating vs Review Count (>=20 reviews)', alpha=0.4)
plt.show()
# plog out the result
!pip install xlsxwriter
out_xlsx = "sg_cafes_quick_report.xlsx"
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    pd.DataFrame({
        "total_rows":[len(df)],
        "with_rating":[df['rating'].notna().sum()],
        "avg_rating_all":[df['rating'].mean()],
        "median_rating_all":[df['rating'].median()],
        "median_review_count_all":[df['userRatingCount'].median()],
        "rows_>=20_reviews":[len(df20)],
        "avg_rating_>=20_reviews":[df20['rating'].mean()],
        "median_review_count_>=20_reviews":[df20['userRatingCount'].median()],
    }).to_excel(writer, index=False, sheet_name="overview")
    df.to_excel(writer, index=False, sheet_name="raw_export")
    df20.to_excel(writer, index=False, sheet_name="filtered_20+reviews")
    top25.to_excel(writer, index=False, sheet_name="top25_filtered")
    chains.to_excel(writer, index=False, sheet_name="name_multiplicity")
print("Prompted to export the finished:", out_xlsx)
import os
os.listdir()
