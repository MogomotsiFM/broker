import boto3
import googlemaps
import json
import requests
import time

from googlemaps.places import places

from botocore.exceptions import ClientError

def get_secret():
    secret_name = "automarket/google_maps"
    region_name = "us-east-1"
    profile_name = "Admin"

    # Create a Secrets Manager client
    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(response["SecretString"])
    return secret["api_key"]


def extractRequiredFields(locationResult):
    requiredFields = {}
    requiredFields["geometry"] = locationResult["geometry"]
    requiredFields["name"]     = locationResult["name"]
    requiredFields["types"]    = locationResult["types"]

    return requiredFields


def isRequiredLocationType(location):
    requiredLocationTypes = ("sublocality", "sublocality_level_1", "neighborhood")
    inter = set(location["types"]).intersection(requiredLocationTypes)

    return len(inter) > 0


def search(api_key, next_page_token=None):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    if next_page_token is None:
        params = f"query=suburbs&location=-25.864573%2C25.620363&radius=50000&key={api_key}"
    else:
        params = f"key={api_key}&pagetoken={next_page_token}"

    response = requests.get(url, params=params)

    return response.json()


# Google Maps API key
api_key = get_secret()
print(f"API Key: {api_key}")

#url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
#params = f"query=suburbs%20in%20mafikeng&radius=20000&key={api_key}"
#
#response = requests.get(url, params=params)
## Returns a next_page_token as part of the response. So fetch until it is empty or None
## Filter the results on type = sublocality_1 | neighborhood
## The results also include the center of an area. Use these to find areas that are at most 2km from a given area.
#
#
#if response.status_code == 200:
#    response = response.json()
#    if response["status"] == "OK":
#        data = response["results"]
#        with open("requests.txt", 'w', encoding='utf-16le') as pg:
#            pg.write("url\n")
#            json.dump(response, pg)
#        
#        # Extract the data we need
#        response["results"] = [extractRequiredFields(result) for result in response["results"] if isRequiredLocationType(result)]
#
#        with open("resultsWithRequiredFields.txt", "w", encoding="utf-16le") as pg:
#            pg.write("url\n")
#            json.dump(response, pg)
#
#        # for d in data:
#        #    d = json.loads(d)
#        #    print(f"Description: {d['description']}    :    {d['structured_formatting']['main_text']}")
#    else:
#        print(f"Request failed: {response['error_message']}")
#else:
#  print(f"Request failed with status: {response.status_code}")


# Using the Google Maps SDK
# maps_client = googlemaps.Client(api_key)
# r2 = places(maps_client, "suburbs in mafikeng", radius=50000)
r2 = search(api_key)
predictions = r2["results"]

with open("requests.txt", 'a', encoding='utf-16le') as pg:
    pg.write("\n\n\nsdk")
    pg.write(json.dumps(r2))


 # Extract the data we need
r2["results"] = [extractRequiredFields(result) for result in r2["results"] if isRequiredLocationType(result)]

areas = []

number = 1
while True:
    print(f"Page: {number}")
    for location in r2["results"]:
        areas.append(location["name"])
        print(location["name"])

    next_page_token = r2.setdefault("next_page_token", None)

    if next_page_token is None:
        break

    number = number + 1
    time.sleep(10)
    try:
        # r2 = places(maps_client, "suburbs in mafikeng", page_token=next_page_token)
        r2 = search(api_key, next_page_token)
        r2["results"] = [extractRequiredFields(result) for result in r2["results"] if isRequiredLocationType(result)]
    except:
        print("Google maps location fetch failed.\n")
        break

with open("google_maps_areas.txt", "a", encoding="utf-16le") as pg:
    areas = set(areas)

    (pg.write(f"{area}\n") for area in areas)
