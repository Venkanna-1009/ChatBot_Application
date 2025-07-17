from bs4 import BeautifulSoup
import requests
import re
import nltk
from urllib.parse import urljoin, urlparse

nltk.download('stopwords')
from nltk.corpus import stopwords


def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        return response.text
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['nav', 'footer', 'header', 'script', 'style']):
        tag.decompose()

    main_content = soup.find('main')
    content_root = main_content if main_content else (soup.body if soup.body else soup)
    paragraphs = [p.get_text(separator=' ', strip=True) for p in content_root.find_all('p') if p.get_text(strip=True)]
    sections = []
    for heading in content_root.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        section = {'heading': heading.get_text(strip=True), 'paragraphs': []}
        for sib in heading.find_next_siblings():
            if sib.name and sib.name.startswith('h') and len(sib.name) == 2:
                break
            if sib.name == 'p':
                para_text = sib.get_text(separator=' ', strip=True)
                if para_text:
                    section['paragraphs'].append(para_text)
        if section['paragraphs']:
            sections.append(section)
    return {'paragraphs': paragraphs, 'sections': sections}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def crawl_and_scrape(base_url):
    html_content = scrape_website(base_url)
    if not html_content:
        print(f"Failed to retrieve HTML content from {base_url}")
        return
    data = parse_html(html_content)
    visited = set()
    visited.add(base_url)
    domain = urlparse(base_url).netloc
    links = set()
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup.find_all('a', href=True):
        abs_link = urljoin(base_url, tag['href'])
        if urlparse(abs_link).netloc == domain and abs_link not in visited:
            links.add(abs_link)
    all_data = [data]
    for link in links:
        visited.add(link)
        page_html = scrape_website(link)
        if page_html:
            page_data = parse_html(page_html)
            all_data.append(page_data)
    with open('extracted_structured_data.txt', 'w', encoding='utf-8') as f:
        for page in all_data:
            for section in page['sections']:
                f.write(f"# {section['heading']}\n")
                for para in section['paragraphs']:
                    f.write(clean_text(para) + '\n')
            for para in page['paragraphs']:
                f.write(clean_text(para) + '\n')
    print('Structured, cleaned data from all crawled pages saved to extracted_structured_data.txt')

def main():
    url = "https://jaajitech.com/"
    crawl_and_scrape(url)

if __name__ == "__main__":
    main()










