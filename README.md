# broker
Automatically matches prospective tenants and property owners.

## Post
Each post contains the location, property type and price

## Intent recognition
Determine if a post is from a property owner or a prospective tenant

## Extract features
Extract the location, property type and price from the post

## Features
### Location
1. If the post is from a service provider then the location is a single point in an area
2. If the post is from a prospective tenant the location may be:
  a. A single area,
  b. A list of areas,
  c. An area and a range (Looking for a place around Montshioa, looking for a place near Mega City)

### Property type
1. Commune,
2. Room,
3. Bachelor(a room with a kitchenette and in suite bathroom),
4. 1 bedroom,
5. 2 bedroom,
6. 3 bedroom,
7. 4 bedroom

### Price
1. If the post is from a service provider then the price is a fixed value,
2. If the post is from a prospective tenant then the price is a range.
