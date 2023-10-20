import os

import pandas as pd

def read_data(posts_file_name):
    with open(posts_file_name, 'r', encoding='utf-16le') as file:
        ds = file.readlines()

    data = [d.strip().lower() for d in ds if d != '\n']

    return data


def label(post):
    """ Rudimentary labelling
        Look for cues
            accommodation + looking + place + need => tenant
            accommodation + available => landlord
            transport => transport
    """
    set_post = set(post.split(' '))

    tenant_cues = set(['look', 'looking', 'find', 'any', 'need'])
    landlord_cues = set(['available'])

    if len(set_post.intersection(tenant_cues)) > 0 and not 'trans' in post:
        return 'tenant'
    if len(set_post.intersection(landlord_cues)) > 0 and not 'trans' in post:
        return 'landlord'

    return 'irrelevant'

data_dir = '..\\data'

# Read in the data
posts = read_data(data_dir + '\\training_data.txt')

labels = [label(p) for p in posts]

df = pd.DataFrame({'posts': posts, 'label': labels})

# Write the data to file
# df.to_csv(data_dir + '\\labeled_intent_detection_dataset.csv')
df.to_csv(os.path.join('..', 'data', 'labeled_intent_detection_dataset.csv'))