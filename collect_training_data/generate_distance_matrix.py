# Read area_names_plus_location.txt file

# Augment the GPS location data with data from hard_coded_area_locations.txt file

# Compute the pairwise distances between areas

# Save all distances less than 2km to file
# File format:
# Each record is a json document
# { 
#       "area": "area_name",
#       "neighbours": ["area_name1", "area_name2", ..., "area_name_n"]
# }

import itertools
import json
import os

from math import radians, cos, sin, asin, sqrt

def distance(start_gps, end_gps):
    lat1 = radians( start_gps["lat"] )
    lon1 = radians( start_gps["lng"] )

    lat2 = radians( end_gps["lat"] )
    lon2 = radians( end_gps["lng"] )

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin( sqrt(a) ) 
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)


def center(google_map_geometry):
    area = google_map_geometry

    ne = area["geometry"]["viewport"]["northeast"]
    sw = area["geometry"]["viewport"]["southwest"]

    lat = ( ne["lat"] + sw["lat"] ) / 2.0
    lng = ( ne["lng"] + sw["lng"] ) / 2.0

    return {"lat": lat, "lng": lng}


locations = []

with open(os.path.join("..", "data", "area_names_plus_location2.txt"), "r") as input:
    locations = input.readlines()

locations = [loc.rstrip() for loc in locations if len(loc) > 0]

matrix = dict()
# Compute distances between pairs of areas
for start_area, end_area in itertools.combinations(locations, 2):
    start_area = json.loads(start_area)
    end_area   = json.loads(end_area)

    start_gps = center(start_area)
    end_gps   = center(end_area)

    dist = distance(start_gps, end_gps)
    print(f"[{start_area['area']}, {end_area['area']}]: {dist}")
    if dist <= 2.25:
        # We want to sort the neighbours by distance
        neigh = {"area": end_area["area"], "distance": dist, "types": end_area["types"]}
        inter = set(end_area["types"]).intersection(["shopping_mall", "school", "university", "church"])
        if len(inter) == 0:
            # Notice how we jump through hoops to add a dictionary to a set.
            matrix.setdefault(start_area["area"], set()).add( json.dumps(neigh) )

        # We want to sort the neighbours by distance
        neigh = {"area": start_area["area"], "distance": dist}
        inter = set(start_area["types"]).intersection(["shopping_mall", "school", "university", "church"])
        if len(inter) == 0:
            matrix.setdefault(end_area["area"], set()).add( json.dumps(neigh) )


print("\n\nWriting matrix to file.")
sorted_matrix = list()
for area, neigh in matrix.items():
    out = dict()
    out["area"] = area
    out["neighbours"] = sorted( list(neigh), key=lambda n: json.loads(n)["distance"] )

    sorted_matrix.append(out)

# Sort by area
sorted_matrix = sorted(sorted_matrix, key=lambda a: a["area"])

with open( os.path.join("..", "data", "distance_matrix.txt"), "w") as output:
    json.dump(sorted_matrix, output, indent=2)

