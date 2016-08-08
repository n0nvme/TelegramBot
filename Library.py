import requests

token = '227495190:AAEu9wUQ0lMFPHOQHs7BNuUA8tyDyVCm7wM'

url = 'https://api.telegram.org/bot' + token


def update(offsetkey):
    offsetkey = str(offsetkey)
    result = requests.get(url + '/getupdates?offset=' + offsetkey)

    result = result.json()

    return result


def send(id, text):

    requests.post(url + '/sendMessage?chat_id=' + id + '&text=' + text)



def get_text():

    text = open('Texts.txt', 'r')

    return text.read()

def run_command(text, id):

    if text == 'start':
        send(id, 'Привет, не хочешь почитать анекдотов моего собсвтенного сочинения?')
        return 0

    elif text == 'help':

        send(id, 'Я - бот, который пишет анекдоты\nНапиши мне "/get", чтобы получить анекдот!')
        return 0

    elif text == 'get':

        aneks = open('Texts.txt', 'r')

        send(id, aneks.read())

        aneks.close()

        return 0
    else:
        return 1