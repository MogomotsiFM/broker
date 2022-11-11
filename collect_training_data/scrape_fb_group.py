import sys
import time

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

'''
Without this import the pyppeteer package install the wrong version of 
chromium in the wrong directory.
shorturl.at/jtx13
'''
import pyppdf.patch_pyppeteer

from requests_html import HTML
from requests_html import HTMLSession


def extract_replies(session, replies_url):
    r = session.get(replies_url)
    time.sleep(15)

    replies = r.html.find('div._2a_i')

    list_replies = []

    for reply in replies:
        comments = reply.xpath('//*/div[2]/div[1]/div/div[2]/text()')

        if len(comments) > 0:
            comment = ' '.join(comments)
            # print(f'Reply: {comment}\n')
            list_replies.append(comment)
        else:
            try:
                # print("Could be an image. Facebook does text extraction.")
                img_tag = reply.find('a img', first=True)
                comment = img_tag.attrs['alt'].split('"')[1]

                # print(f'Reply: {comment}\n')
                list_replies.append(comment)
            except Exception:
                print(f'\n{replies_url}')
                # print('Fine, we can get more data elsewhere.')

    return list_replies


def extract_data(html):
    reply_urls = []
    posts = []
    num_comments = []

    articles = get_article(html)
    
    for article in articles:
        try:
            username = article.find('div._4g34 a', first=True).text
            print(f'User: {username}')

            comments = article.find('div > span > p')
            comment = '\n'.join([c.text for c in comments])
            comment = ' '.join(comment.split('\n'))

            if len(comments) > 0:
                posts.append(comment)
            else:
                try:
                    comment = article.find('span._2z7d', first=True).find('span', first=True).text

                    comment = ' '.join(comment.split('\n'))
                    posts.append(comment)
                except Exception:
                    print('We can get more data elsewhere!')

            # Are there replies to this comment
            num_replies = article.find('footer', first=True).find('div._1fnt > span', first=True)
            try:
                if num_replies.text:
                    replies_url = article.find('footer', first=True).find('a', first=True).attrs['href']
                    reply_urls.append(replies_url.strip())

                    num_replies = num_replies.text.split(' ')[0]
                    num_comments.append(int(num_replies))
            except AttributeError:
                # There are no comments, swallow this exception
                pass
            except Exception as exc:
                print(exc)

            time.sleep(1)
        except Exception as exc:
            print('Something else went wrong...')
            print(exc)

    return reply_urls, posts, num_comments


def get_article(html):
    index = 0
    not_found_count = 0
    while True:
        article = html.xpath(f'//*[@data-store-id="{index}"]', first=True)

        if article is not None:
            # Make sure it is the article we need
            article_class = article.xpath('//*[@class="_55wo _5rgr _5gh8 async_like"]', first=True)
            if article_class is not None:
                not_found_count = 0
                yield article
        else:
            not_found_count = not_found_count + 1
            if not_found_count > 1000:
                print('Have not seen an article in a while.')
                break
        index = index + 1


def scroll(driver, page_load_time, max_scrolls):
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while max_scrolls > 0:
        print(f'Scroll: {max_scrolls}')
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(page_load_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # If heights are the same it will exit the function
            break
        last_height = new_height

        max_scrolls = max_scrolls - 1


def log_in(driver, e_mail, password):
    email_input = driver.find_element(by=By.XPATH, value='//*[@id="m_login_email"]')
    email_input.send_keys(e_mail)
    try:
        driver.find_element(by=By.XPATH, value='//*[@id="password_input_with_placeholder"]/input').send_keys(password)
    except NoSuchElementException:
        driver.find_element(by=By.XPATH, value='//*[@id="m_login_password"]').send_keys(password)

    try:
        driver.find_element(by=By.XPATH, value='//*[@id="login_form"]/ul/li[3]/input').send_keys(Keys.ENTER)
    except NoSuchElementException:
        driver.find_element(by=By.XPATH, value='//*[@id="login_password_step_element"]/button').send_keys(Keys.ENTER)


def fetch_fb_group_data(e_mail, password, group_id):
    URL = "https://m.facebook.com/groups/" + GROUP_ID # + "/?sorting_setting=CHRONOLOGICAL"

    # Setup the driver. This one uses firefox with some options and a path to the geckodriver
    driver = webdriver.Chrome()#Firefox(options=options, executable_path='./geckodriver')
    # implicitly_wait tells the driver to wait before throwing an exception
    page_load_time = 60
    driver.implicitly_wait(60)
    # driver.set_page_load_timeout(page_load_time)
    # driver.set_script_timeout(page_load_time)

    # driver.get(url) opens the page
    driver.get(f'https://m.facebook.com/login.php?next=https%3A%2F%2Fm.facebook.com%2Fgroups%2F{group_id}')

    time.sleep(60)

    # Log into facebook
    log_in(driver, e_mail, password)

    time.sleep(60)

    # This starts the scrolling by passing the driver and a timeout
    scroll(driver, 30, 1500)
    # Once scroll returns bs4 parsers the page_source
    # soup_a = BeautifulSoup(driver.page_source, 'lxml')

    html = HTML(html=driver.page_source)

    driver.close()

    return html


if __name__ == '__main__':
    """
    Usage: python3 scrape_fb_group.py fb_username fb_password
    """

    if len(sys.argv) == 3:
        GROUP_ID = "353422322429171"
        e_mail = sys.argv[1]
        password = sys.argv[2]
        print(f'Username: {e_mail}    password: {password}')

        html = fetch_fb_group_data(e_mail, password, GROUP_ID)

        with open('scrolled_fb2.html', 'w', encoding='utf-16le') as pg:
            pg.write(html.html)
    else:
        with open('scrolled_fb.html', 'r', encoding='utf-16le') as file:
            data = file.read()
            html = HTML(html=data)


    session = HTMLSession()

    formater = lambda comment: ' '.join( comment.split('\n') ) + '\n'

    with open('training_data.txt', 'w', encoding='utf-16le') as file:
        reply_urls, posts, num_replies = extract_data(html)
        print(f'Posts: {len(posts)}    {len(reply_urls)}    {len(num_replies)}')

        file.writelines([p+'\n\n' for p in posts])

        with open('replies_urls.txt', 'w', encoding='utf-16le') as f:
            f.writelines([url+'\n\n' for url in reply_urls])

        for count, url in zip(num_replies, reply_urls):
            print('Another one...')
            replies = extract_replies(session, url)

            file.writelines([r+'\n\n' for r in replies])

            # Sleep a time that is proportional to the number of comments
            time.sleep(count/10)
