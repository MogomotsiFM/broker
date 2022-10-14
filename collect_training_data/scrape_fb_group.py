import time
import random
from time import sleep

'''
Without this import the pyppeteer package install the wrong version of 
chromium in the wrong directory.
shorturl.at/jtx13
'''
import pyppdf.patch_pyppeteer

from requests_html import HTML
from requests_html import HTMLSession


'''
class FBGroupScraper:

    def __init__(self, group_id):
        self.group_id = group_id
        self.page_url = "https://www.facebook.com/groups/" + self.group_id
        self.page_content = ""

    def get_page_content(self):
        self.page_content = requests.get(self.page_url).text

    def parse(self):
        soup = BeautifulSoup(self.page_content, "html.parser")
        feed_container = soup.find_all(string=re.compile("m8h3af8h l7ghb35v kjdc1dyq kmwttqpk gh25dzvf")) #.find_all("p")
        for i in feed_container:
            print(i.text)
'''
script = """
    () => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }
"""

def extract_replies(session, replies_url):
    r = session.get(replies_url)

    replies = r.html.find('div._2a_i')

    for reply in replies:
        try:
            tmp = reply.find('div._2b06', first=True)
            #print(tmp.text)
            comment = ' '.join(tmp.text.split('\n')[1:])
            print(f'Reply: {comment}\n')
        except AttributeError:
            print("Could be an image. Facebook does text extraction.")
            img_tag = reply.find('a img', first=True)# .find('i', first=True)
            #print(img_tag.html)
            #img_tag_attrs = img_tag.attrs
            comment = img_tag.attrs['alt'].split('"')[1]
            print(f'Reply: {comment}\n')

        # Does this comment have replies


def extract_data(html):
    reply_urls = []

    articles = html.find("article")
    for article in articles:
        username = article.find('div._4g34 a', first=True).text
        print(f'User: {username}')
        try:
            comment = article.find('p', first=True).text
        except AttributeError:
            comment = article.find('span._2z7d', first=True).find('span', first=True).text
        print(f'Comment: {comment}')

        # Are there replies to this comment
        # These span tag class is obtained from inspecting the html returned by this app and not
        # the one returned directly by the browser. This could mean the is scraper fragile.
        num_replies = article.find('footer', first=True).find('span.cmt_def._28wy', first=True)
        try:
            print(num_replies.text)
            if num_replies.text:
                replies_url = article.find('footer', first=True).find('a', first=True).attrs['href']
                # print(f'Replies: {repr(replies_url)}')
                reply_urls.append(replies_url.strip())
        except AttributeError:
            print('0 comments')

        print()

        time.sleep(1)

    return reply_urls


print(script)

GROUP_ID = "353422322429171"
URL = "https://m.facebook.com/groups/" + GROUP_ID + "?sorting_setting=CHRONOLOGICAL"
session = HTMLSession()
response = session.get(URL)
html = response.html

reply_urls = extract_data(html)

print('\n\n\nPrint a single reply from each post')
#for url in reply_urls:
#    extract_replies(session, url)
#    time.sleep(1)
# Simulate how a person would read the comments: depth first instead of breadth first
while len(reply_urls) != 0:
    extract_replies(session, reply_urls.pop())
    # Sleep a time that is proportional to the number of comments
    time.sleep(1)
