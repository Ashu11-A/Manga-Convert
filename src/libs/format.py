from urllib.parse import urlparse

def format_url(image_url):
    # Remove espaços em branco
    image_url = image_url.strip()

    # Adiciona 'http://' se a URL não tiver um esquema
    if not urlparse(image_url).scheme:
        image_url = 'http://' + image_url

    # Verifica se a URL é válida
    parsed_url = urlparse(image_url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        return None  # Retorna None se a URL não for válida

    return image_url