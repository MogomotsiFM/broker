"""
Reads the list of all areas in Mahikeng,
Uses Google maps API to estimate the GPS location of the area,
Writes the data to file.
Each output is a JSON object of the following format:
 { 
       "area": "area_name", 
       "description": string,
       "location": "viewport as given by GPS coordinates",
       "type": [string],
       "name": "area name as retrieved from Google maps"
 }

 This output file is used in generate_distance_matrix.py to get a list of neighbours within 2.5km.
"""

import boto3
import googlemaps
import json
import os
import re
import time

from googlemaps.places import place
from googlemaps.places import places_autocomplete

from botocore.exceptions import ClientError

def get_area_names():
    area_names = set()
    with open(os.path.join("..", "data", "areas.txt"), "r") as file:
        # Some lines have parenthesis at the end
        for area_name in file.readlines():
            idx = area_name.find('(')
            if idx > 0:
                area_name = area_name[:idx]

            area_names.add(area_name.lower().strip())

    return area_names


def get_google_maps_token():
    region = "us-east-1"
    profile = "automarket_app"

    session = boto3.session.Session(region_name=region, profile_name=profile)
    ssm = session.client(
            #service_name='secretsmanager',
            service_name="ssm",
            region_name=region
        )

    secret_name = "/broker_app/google_maps"

    response = ssm.get_parameter(
        Name=secret_name,
        WithDecryption=True
    )

    return response["Parameter"]["Value"]


def get_candidate_areas(area: str, central_location, radius):
    if area == "mmabana" or area == "Mmabana":
        candidates = places_autocomplete(
            maps_client,
            components={"country": ["ZA"]},
            input_text=area, 
            location=central_location,
            radius=radius,
        )
        for candidate in candidates:
            candidate["type"] = ["locality", "sublocality", "neighborhood"]

        return candidates

    candidates = places_autocomplete(
        maps_client,
        components={"country": ["ZA"]},
        input_text=area, 
        location=central_location,
        radius=radius,
        types="locality|sublocality|sublocality_level_1|neighborhood"
    )

    if ( len(candidates) == 0 ):
        time.sleep(1)
        candidates = places_autocomplete(
            maps_client,
            components={"country": ["ZA"]},
            input_text=area,
            location=central_location,
            radius=radius,
            # We use a hack to get results for Ext 38 and 39. 
            # The routes they are named after pass right through these areas.
            types="shopping_mall|school|university|church|route"
        )

    return candidates


def is_area_in_maf(cand):
    reg_exp = re.compile("(Mahikeng)|(Mmabatho)|(Mafikeng)")

    first  = not reg_exp.search(cand.setdefault("description", "not found")) is None
    second = not reg_exp.search(cand.setdefault("formatted_address", "not found")) is None

    return first or second


# Google Maps API key
api_key = get_google_maps_token()

# Using the Google Maps SDK
maps_client = googlemaps.Client(api_key)

area_names = get_area_names()

# Mafikeng central point
central_location = "-25.864573,25.620363"
radius = 20000

# Extend the area names with location data. 
# This will be used to find areas closest to a given one.
areas_with_location_data = set()

count = 1
for area in area_names:
    candidates = get_candidate_areas(area, central_location, radius)

    # Filter out areas outside Mafikeng and Mmabatho
    if len( candidates ) > 1:
        candidates = [cand for cand in candidates if is_area_in_maf(cand) is True]
        print(f"{area}: Number of candidates: {len(candidates)}")
        # Take the first item only
        # TODO: Find a better way to get to one item
        if len(candidates) > 1:
            candidates = [candidates[0]]
    elif len( candidates ) == 0:
        print(f"{area} not found in google maps")

    # Retrieve the gps location of each of these places  
    for candidate in candidates:
        time.sleep(1)
        response = place(
            maps_client,
            place_id = candidate["place_id"],
            fields=["name","geometry/viewport/northeast","geometry/viewport/southwest", "type"]
        )

        areas_with_location_data.add( json.dumps(
            {
                "area": area, 
                "description": candidate["description"], 
                "geometry": response["result"]["geometry"],
                "types": response["result"]["types"],
                "name": response["result"]["name"]
            }
        ))
        
    print(f"{count}/{len(area_names)} : {area}")
    count = count + 1

    time.sleep(1)

# Generate missing data
#   town == mafikeng
#   stateng == stadt
#   university == north west university
#   setumo park == motswedi street
#   Ext 39 == "Ext 39 Rd"
#   Siga  == "Ext 39 Rd"    // Siga is Ext 39
#   Ext 38 == "One St"

# Encode this in a dictionary
area_mappings = {
    "mafikeng": ["town"],
    "stadt": ["stateng"],
    "north west university": ["university", "NWU"],
    "motswedi street": ["setumo park"],
    "ext 39 rd": ["ext 39", "siga", "ciga"],
    "one st": ["ext 38", "smarties"]
}


with open(os.path.join("..", "data", "area_names_plus_location.txt"), mode="w") as output:
    # [output.write(f"{result}\n") for result in areas_with_location_data]
    for area_with_location in areas_with_location_data:
        jarea = json.loads(area_with_location)

        if not "route" in jarea["types"]:
            output.write(f"{area_with_location}\n")


        mapping = area_mappings.setdefault( jarea["area"], [])
        for area in mapping:
            print(f"Maped {area} to {jarea['area']}")

            jarea["area"] = area
            jarea["name"] = area

            if "route" in jarea["types"]:
                jarea["types"] = ["sublocality", "neighborhood"]

            output.write(f"{json.dumps(jarea)}\n")

