import json
import os
import re
from fuzzywuzzy import process, fuzz

import logging
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd

def get_area_names():
    area_names = set()
    with open(os.path.join('..', 'data', 'areas.txt'), 'r') as file:
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

    return re.compile(bach + '|' + house1 + '|' + house2 + '|' + room + '|house|commune')


def create_contact_type_re():
    # Notice the hack to ensure that the last character is a number and not a space.
    return re.compile(r'(0|\+?[0-9]{1,3})\s?[0-9\s]{8,10}[0-9]')


def replace_location_with_ner_marker2(processed_post, area_names):
    marked_post = []

    words = processed_post.split(' ')

    used_last_word = False
    w2 = ''

    for w1, w2 in zip(words, words[1:]):
        if not used_last_word:
            if not w1.isupper() and len(w1) + len(w2) > 3:
                _, prob = process.extractOne(w1, area_names, scorer=fuzz.ratio)

                _, prob2 = process.extractOne(' '.join([w1, w2]), area_names, scorer=fuzz.token_sort_ratio)

                if prob2 > prob and prob2 > 80 and prob > 20:
                    marked_post.append('P_LOC')

                    used_last_word = True
                elif prob > 80:
                    marked_post.append('P_LOC')

                    used_last_word = False
                else:
                    marked_post.append(w1)

                    used_last_word = False
        
            else:
                marked_post.append(w1)

        else:
            used_last_word = False

    # Evaluate the last word by itself
    if not used_last_word:
        if len(w2) > 3:
            _, prob = process.extractOne(w2, area_names, scorer=fuzz.ratio)
            if prob > 80:
                marked_post.append('P_LOC')
            else:
                marked_post.append(w2)
        else:
            marked_post.append(w2)

    return ' '.join(marked_post)


def replace_location_with_ner_marker(processed_post, area_names):
    words = processed_post.split(' ')
    words = [word.strip() for word in words if not word in ['', ' ']]

    marked_post, _ = replace_location_with_ner_marker_helper(words, area_names)
    
    insert_ner_marker = []
    for mark, word in zip(marked_post, words):
        if mark > 0:
            insert_ner_marker.append(f'P_LOC{mark}')
        else:
            insert_ner_marker.append(word)

    return ' '.join(insert_ner_marker)


def find_area_name(words, start, area_names):
    name = []
    max_quality = -1
    # The next word to read
    word_idx = start
    
    name_start = start

    # Keep adding words as long as the quality of our match is improving
    while word_idx < len(words):
        name.append(words[word_idx])

        _, quality = process.extractOne(' '.join(name), area_names, scorer=fuzz.ratio)
        
        if quality > max_quality:
            max_quality = quality

            word_idx = word_idx + 1
        else:
            name.pop()

            break

    # Keep removing words at the begining as long as the quality of our match is improving
    while len(name) > 0:
        removed_name = name.pop(0)
        name_start = name_start + 1

        _, quality = process.extractOne(' '.join(name), area_names, scorer=fuzz.ratio)
        
        # If we try to remove a word and it does not affect the quality of our match the we remove it.
        if quality == max_quality and len(removed_name) > 1:
            name.insert(0, removed_name)
            name_start = name_start - 1

            break
        if quality >= max_quality:
            max_quality = quality
        else:
            name.insert(0, removed_name)
            name_start = name_start - 1

            break  

    return name, name_start, word_idx


def replace_location_with_ner_marker_helper(list_of_words, area_names):
    marked_posts = []
    found_words = []

    words = list_of_words
    PROB_THRESHOLD = 90

    print(words)
    
    word_idx = 0
    while word_idx < len(words):
        name, name_start, word_end = find_area_name(words, word_idx, area_names)

        found_area_name, prob = process.extractOne(' '.join(name), area_names, scorer=fuzz.ratio)
        
        if prob >= PROB_THRESHOLD:
            for _ in range(word_idx, name_start):
                marked_posts.append(0)
                found_words.append(' ')

            for idx, _ in enumerate(range(name_start, word_end), 1):
                marked_posts.append(idx)
                found_words.append(found_area_name)
        else:
            for _ in range(word_idx, word_end):
                marked_posts.append(0)
                found_words.append(' ')

        word_idx = word_end

    return marked_posts, found_words
    

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
    units_re = re.compile(r'(?P<units>unit\s?[0-9]{1,2}(\s?(,|/|or)\s?[0-9]{1,2})*)')

    # unit 1,2,3 or 6 
    units_match = units_re.search(processed_post)
    if units_match is not None:
        # Split on the separator
        raw_units = processed_post[units_match.start():units_match.end()]

        units, _ = re.compile(separators).subn(lambda _: ' ', raw_units)
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


def process_surounding_areas(processed_post, distance_matrix):
    # near_re = re.compile(r'near((er)? to)?|around|(closer? to)|(walking distance to)')
    near_re = re.compile(r'near(er)? (to\s)?|(close(r)? (to\s)?)|(walking distance (to\s)?)')

    seen = near_re.search(processed_post)

    if seen is not None:
        # Find the area that best matches the 
        # Where to start searching
        start, end = seen.span()
        words = processed_post[end:].strip().lower().split(' ')
        words = [word.strip() for word in words if word != ' ']
        
        marked_words, matched_words = replace_location_with_ner_marker_helper(words, distance_matrix.keys())

        area_name = None
        if marked_words[0] == 'P_LOC':
            area_name = matched_words[0]
        elif len(marked_words) > 1 and marked_words[1]:
            area_name = matched_words[1]
        else:
            print("The post is ill-formed")
            return processed_post
        
        try:
            neighbours = distance_matrix[area_name]

            neighbours = [json.loads(neigh)["area"] for neigh in neighbours]

            return processed_post[:start] + ' in ' + ' , '.join(neighbours) + ' or ' + processed_post[end:]
        except:
            with open(os.path.join("..", "data", "unknown_areas.txt"), "w") as output:
                output.write(area_name)

            print(f"Area not found in distance matrix: {area_name}")
            return processed_post

    return processed_post



def label_posts(posts):
    processed_posts = []

    area_names = get_area_names()
    distance_matrix = load_distance_matrix()

    i = 1
    for _, row in posts.iterrows():
        post = row["posts"]
        label = row["label"]

        print(f'    {i}: {post} : {label}')
        i += 1

        try:
            processed_post    = preprocess(post)
            processed_post, _ = property_re.subn('P_TYPE', processed_post, 100)
            processed_post, _ = price_re.subn('P_PRICE', processed_post, 100)
            processed_post, _ = contacts_re.subn('P_TELL', processed_post, 100)

            processed_post = expand_units(processed_post)
            processed_post, _ = re.compile(r'[.,]').subn(lambda m: ' ' + processed_post[m.start():m.end()] + ' ', processed_post)
            if label == "tenant":
                processed_post = process_surounding_areas(processed_post, distance_matrix)
            
            print(processed_post)
            processed_post = replace_location_with_ner_marker(processed_post, area_names)

            print(processed_post)
            processed_posts.append(processed_post+'\n')
        except Exception as exp:
            print(post)
            print(exp)

    return processed_posts


def label_posts_for_ner_task(input_filename, output_filename):
    # with open(input_filename, 'r', encoding='utf-16le') as file:
    #    posts = file.readlines()
    posts = pd.read_csv(input_filename)

    processed_posts = label_posts(posts)

    with open(output_filename, 'w', encoding='utf-16le') as file:
        file.writelines(processed_posts)


def load_distance_matrix():
    # Load distance matrix 
    matrix = dict()
    with open(os.path.join("..", "data", "distance_matrix.txt"), "r") as input:
        print("        Loading distance matrix data...")
        matrix = json.load(input)

    mat = dict()
    for item in matrix:
        mat[item["area"]] = item["neighbours"]

    return mat


if __name__ == '__main__':
    area_names = get_area_names()

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

    post = "Um looking for a place to rent.. Any recommendations near Lomanyaneng?"
    print('\n\n', f"Post: {post}")
    processed_post = process_surounding_areas(post, load_distance_matrix())
    print(processed_post)

    post = "hi im looking for a room nearer unit 14 . my budget is 1800"
    print('\n\n', f"Post: {post}")
    processed_post = process_surounding_areas(post, load_distance_matrix())
    print(processed_post)

    post = "hi im looking for a room nearer to unit 14 . my budget is 1800"
    print('\n\n', f"Post: {post}")
    processed_post = process_surounding_areas(post, load_distance_matrix())
    print(processed_post)

    post = "hi im looking for a room close to mega city . my budget is 1800"
    print('\n\n', f"Post: {post}")
    processed_post = process_surounding_areas(post, load_distance_matrix())
    print(processed_post)

    post = "hi im looking for a room closer to mega city . my budget is 1800"
    print('\n\n', f"Post: {post}")
    processed_post = process_surounding_areas(post, load_distance_matrix())
    print(processed_post)

    label_posts_for_ner_task(
        #os.path.join('..', 'data', 'training_data.txt'), 
        os.path.join('..', 'data', 'labeled_intent_detection_dataset.csv'),
        os.path.join('..', 'data', 'labelled_ner_training_data.txt'))
