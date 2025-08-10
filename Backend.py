import requests

def download_pdf(url, filename="temp.pdf"):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename
