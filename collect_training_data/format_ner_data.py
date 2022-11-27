import re
from fuzzywuzzy import process, fuzz

def get_area_names():
    area_names = set()
    with open('areas.txt', 'r') as file:
        # Some lines have parenthesis at the end
        for area_name in file.readlines():
            idx = area_name.find('(')
            if idx > 0:
                area_name = area_name[:idx]

            area_names.add(area_name.lower().strip())

    return area_names


def create_price_re():
    """ Creates price regular expression used to search for the 
    cost of the accomodation in a post.

    Price: R1800(.00) | R1.800(.00) | R1.8k
    """
    price_re1 = r'\b[Rr]?[1-9][0-9]{2,}(\.[0-9]{2,2})?\b'
    price_re2 = r'\b[Rr]?[1-9],[0-9]{3,}(\.[0-9]{2,2})?\b'
    price_re3 = r'\b[Rr]?[1-9](\.[1-9])?k\b'

    return re.compile(price_re1 + '|' + price_re2 + '|' + price_re3)


def create_property_type_re():
    bach   = r'bachelor(\srooms?)?'
    house1 = r'[0-9]+(\s?or\s?[0-9])?\s*bed\s*rooms*(\shouse)?'
    house2 = r'[0-9](\s?or\s?[0-9])?\s?roomed[-\s]house'
    room   = r'\b((1|one|single)\s?)?rooms?'

    return re.compile(bach + '|' + house1 + '|' + house2 + '|' + room + '|house')


def create_contact_type_re():
    # Notice the hack to ensure that the last character is a number and not a space.
    return re.compile(r'(0|\+?[0-9]{1,3})\s?[0-9\s]{8,10}[0-9]')


def find_start(processed_post):
    yield 0
    for i, c in enumerate(processed_post):
        if c == ' ':
            yield i+1
    yield len(processed_post) + 1


def find_location_name(processed_post, area_names):
    word_bounds = list(find_start(processed_post))

    area_names_boundaries = []

    # We assume that each area name may have at most two words
    for start1, start2, end in zip(word_bounds, word_bounds[1:], word_bounds[2:]):
        w1 = processed_post[start1:start2-1]
        w2 = processed_post[start2:end-1]
        if not w1.isupper() and len(w1) + len(w2) > 3:
            word, prob = process.extractOne(w1, area_names, scorer=fuzz.ratio)

            word2, prob2 = process.extractOne(' '.join([w1, w2]), area_names, scorer=fuzz.token_sort_ratio)

            print(w1, ' : ', w2, ' : ', word, ' : ', word2, ' : ', prob, ' : ', prob2)
            if prob2 > prob and prob2 > 90:
                new_bounds = (start1, end-1, prob2)
                settle_possible_overlap(area_names_boundaries, new_bounds)

            if prob > 90:
                # area_names_boundaries.append((start1, start2-1, prob))
                new_bounds = (start1, start2-1, prob)
                settle_possible_overlap(area_names_boundaries, new_bounds)

    return area_names_boundaries


def settle_possible_overlap(area_names_boundaries, new_bound):
    if len(area_names_boundaries) == 0:
        area_names_boundaries.append(new_bound)
        return

    last_start, last_end, last_prob = area_names_boundaries.pop()
    start, end, prob = new_bound

    # The two boundaries overlap
    if last_start < start < last_end:
        if last_prob > prob:
            area_names_boundaries.append((last_start, last_end, prob))
        else:
            area_names_boundaries.append((start, end, prob))
    else:
        area_names_boundaries.append((last_start, last_end, last_prob))
        area_names_boundaries.append((start, end, prob))


def replace_location_with_ner_marker(processed_post: str, area_names):
    area_names_boundaries = find_location_name(processed_post, area_names)

    # As we keep replacing longer area names with shorter NER label, the
    # entire list is essentially shifted downwards, hence the offset.
    offset = 0
    for start, end, _ in area_names_boundaries:
        old = processed_post[start-offset:end-offset]
        processed_post = processed_post.replace(old, 'P_LOC')

        offset = offset + (end - start) - len('P_LOC')

    return processed_post


def preprocess(post):
    processed_post = post.strip().lower()
    # Add spaces around puctuation signs
    add_spaces = lambda ms: ' ' + processed_post[ms.start():ms.end()] + ' '
    processed_post, _ = re.compile(r'[-!&*\(\)/]').subn(add_spaces, processed_post, 100)

    processed_post, _ = re.compile(r'\s{2,}').subn(lambda x: ' ', processed_post, 100)

    return processed_post


def expand_units(processed_post: str):
    separators = r'(,|/|or)'
    # Mmabatho units re: unit 1, 2, 3 or 6
    # units_re = re.compile(r'unit\s?[0-9]{1,2}(\s?[,/]\s?[0-9]{1,2})*(\s?or\s?[0-9]{1,2})?')
    units_re = re.compile(r'unit\s?[0-9]{1,2}(\s?(,|/|or)\s?[0-9]{1,2})*')

    # unit 1,2,3 or 6 
    units_match = units_re.search(processed_post)
    if units_match is not None:
        # Split on the separator
        raw_units = processed_post[units_match.start():units_match.end()]

        units, _ = re.compile(separators).subn(lambda m: ' ', raw_units)
        units = units.split()

        areas = ''
        root = units[0]
        for u in units[1:]:
            if areas == '':
                areas = root + ' ' + u
            else:
                areas = areas + ', ' + root + ' ' + u

        processed_post = processed_post.replace(raw_units, areas)
    return processed_post


def near(area):
    """ Get a list of areas within 2km of the given area.
    We need an API for this.
    We may even use the API to get a more complete list of the areas around Mafikeng.
    """
    return [a+f'{i}' for i, a in enumerate(area) if a != ' ']


def process_surounding_areas(processed_post):
    near_re = re.compile(r'near|around|(close to)|(walking distance to)')

    seen = near_re.search(processed_post)

    if seen is not None:
        pass

    return processed_post


area_names = get_area_names()

# with open('areas3.txt', 'w') as file:
#    for area in area_names:
#        file.write(f'{area}\n')


# Property types
property_re = create_property_type_re()
price_re    = create_price_re()
contacts_re = create_contact_type_re()

ss = 'single room available to rent next to the ext 39 for r1.5k call or watsapp 0734872827'
print(property_re.search(ss))
print(process.extractOne(ss, area_names))
print(process.extractOne(ss, ['ext 39']))
print(price_re.search(ss))
ss = 'Any accomodations near mgacity,male single room for next year'
print(process.extractOne(ss, area_names))

us1 = 'hii, am in need of a P_TYPE to rent that costs atleast 3000 a month around unit 2,7,8,9 or 12'
print(us1)
print(expand_units(us1))
us2 = 'hii, am in need of a P_TYPE to rent that costs atleast 3000 a month around unit 2/7/8/9 or 12'
print(us2)
print(expand_units(us2))

# ss = 'Room available to rent in lonely park next to the crossing for R1500 call or watsapp 0734872827'
# ss = 'Dumelang Im looking for a bachelor, preferably ko di units or montshioa for next month...Thank you.'
# ss = 'Room available to rent in lonely park or (signalhill next to the crossing for R1500 call or watsapp 0734872827'
ss = 'Bachelor rooms for rent ( Signalhill- Mafhikeng ) Rent; R1,800.00 Watsup: 0633312221'
# a  = 'P_TYPE for rent ( Signalhill- Mafhikeng ) Rent; P_PRICE Watsup: 0633312221'
# b  = '0      7   11  16 19         30        39 41   47      55       63         73'
ss = "Hello I'm Looking For A Room To Rent Preferably At Imperial Reserve Or Golf View. My Budget Is R1500 to R2000"
w, p = process.extractOne(ss, area_names)
print(ss)
processed_post = preprocess(ss)
processed_post, _ = property_re.subn('P_TYPE', processed_post, 100)
print(processed_post)
processed_post, _ = price_re.subn('P_PRICE', processed_post, 100)
print(processed_post)
print('Here\n')
processed_post = replace_location_with_ner_marker(processed_post, area_names)
print(processed_post)

print('\n\nHere')
processed_post = replace_location_with_ner_marker("hi i'm looking 4 a room mo majemantsho", area_names)
print(processed_post)


with open('training_data.txt', 'r', encoding='utf-16le') as file:
    posts = file.readlines()

i = 0
with open('labeled_ner_training_data.txt', 'w', encoding='utf-16le') as file:
    for post in [p for p in posts[0:1] if len(p) > 2]:
        print(f'    {i}')
        i += 1
        processed_post    = preprocess(post)
        processed_post, _ = property_re.subn('P_TYPE', processed_post, 100)
        processed_post, _ = price_re.subn('P_PRICE', processed_post, 100)
        processed_post, _ = contacts_re.subn('P_TELL', processed_post, 100)

        processed_post = expand_units(processed_post)
        processed_post, _ = re.compile(r',').subn(lambda m: ' , ', processed_post)
        processed_post = replace_location_with_ner_marker(processed_post, area_names)

        file.write(f'{post}')
        file.write(f'{processed_post}\n\n')
