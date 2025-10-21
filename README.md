pip install requests pandas
# ---- SUPER SIMPLE Google Places v1 crawler for Singapore cafes ----
import time, math, requests
import pandas as pd

# ====== 1) ENTER YOUR API KEY HERE ======
API_KEY = "YOUR_API_KEY_HERE"  # ⚠️ Do NOT upload your real key to GitHub
assert API_KEY, "Please fill in your Google Maps API Key above."

# ====== 2) BASIC SETTINGS ======
NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.rating",
    "places.userRatingCount",
    "places.types",
    "places.location",
    "places.priceLevel",
    "places.googleMapsUri"
])
HEADERS = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": FIELD_MASK
}

# Approximate bounding box of Singapore
MIN_LAT, MAX_LAT = 1.200, 1.470
MIN_LNG, MAX_LNG = 103.600, 104.100

# Search radius in meters
RADIUS_M = 1800

# For beginners: set TEST_MODE = True to fetch only a few grid points first
TEST_MODE = False   # True = small test run; False = full scan

# ====== 3) Generate grid points to cover Singapore ======
LAT_M_PER_DEG = 111_320
LNG_M_PER_DEG = math.cos(math.radians(1.35)) * 111_320
lat_step = (RADIUS_M * 1.6) / LAT_M_PER_DEG
lng_step = (RADIUS_M * 1.6) / LNG_M_PER_DEG

def gen_grid():
    """Generate a set of latitude/longitude centers covering Singapore."""
    lats, lngs = [], []
    cur = MIN_LAT
    while cur <= MAX_LAT:
        lats.append(round(cur, 6)); cur += lat_step
    cur = MIN_LNG
    while cur <= MAX_LNG:
        lngs.append(round(cur, 6)); cur += lng_step

    centers = []
    for i, la in enumerate(lats):
        off = 0.0 if i % 2 == 0 else lng_step / 2  # Shift every other row (hex-like)
        for lo in (x + off for x in lngs):
            if lo <= MAX_LNG:
                centers.append((la, round(lo, 6)))
    if TEST_MODE:
        # Limit to first few points for testing
        centers = centers[:6]
    return centers

centers = gen_grid()
print(f"[INFO] grid centers: {len(centers)} (TEST_MODE={TEST_MODE})")

# ====== 4) Function to call the API near each center ======
def search_nearby_cafes(lat, lng, radius_m=RADIUS_M, max_pages=3):
    """
    Search for cafes around one center point using Google Places API (v1).
    Each call may return multiple pages of up to 20 results.
    """
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
            time.sleep(1.2)  # Wait for the next page token to activate
        r = requests.post(NEARBY_URL, headers=HEADERS, json=body, timeout=30)
        if r.status_code != 200:
            print("[WARN] HTTP", r.status_code, "->", r.text[:200])
            break
        j = r.json()
        results += j.get("places", [])
        page_token = j.get("nextPageToken")
        if not page_token:
            break
    return results

# ====== 5) Main loop: fetch data for all grid points ======
all_rows = {}   # Use place_id as unique key to avoid duplicates
errors = 0

for (la, lo) in centers:
    try:
        places = search_nearby_cafes(la, lo, max_pages=3)
        for p in places:
            pid = p.get("id")
            if not pid or pid in all_rows:
                continue
            row = {
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
            all_rows[pid] = row
    except Exception as e:
        errors += 1
        print("[WARN] error at center:", (la, lo), "->", repr(e))

print(f"[INFO] unique cafes: {len(all_rows)} | errors: {errors}")

# ====== 6) Save results to CSV ======
df = pd.DataFrame.from_dict(all_rows, orient="index").reset_index(drop=True)
out_csv = "sg_cafes_places_v1.csv"
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"[DONE] saved {df.shape[0]} rows -> {out_csv}")

# ====== 7) Quick summary  ======
print("Total rows:", len(df))
print("With rating:", df["rating"].notna().sum())
print("Median review count:", df["userRatingCount"].median())

# ====== 8) plot  ======
import matplotlib.pyplot as plt

# First, create the df20 dataframe by filtering cafes with ≥20 reviews
df20 = df[df['userRatingCount'] >= 20]  # This line creates the df20 dataframe

# Rating distribution (all cafes)
plt.figure()
df['rating'].dropna().plot(kind='hist', bins=20, title='Rating Distribution (All Cafes)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Rating distribution (cafes with ≥20 reviews)
plt.figure()
df20['rating'].dropna().plot(kind='hist', bins=20, title='Rating Distribution (≥20 Reviews)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Rating vs. number of reviews (cafes with ≥20 reviews)
plt.figure()
df20.plot(kind='scatter',
          x='userRatingCount',
          y='rating',
          title='Rating vs. Review Count (≥20 Reviews)',
          alpha=0.4)
plt.show()
