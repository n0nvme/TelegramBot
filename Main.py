import Library



update_id = Library.update(1)['result'][0]['update_id']
a = update_id
b = 1

while 1:

    try:

        update = Library.update(-1)



        if update_id != update['result'][0]['update_id']:

            print(b)
            log_file = open('Log.txt', 'a')
            log_file.write(str(b) + ' ' + str(update) + '\n')
            if update['result'][0]['update_id'] - a != 1:
                loss = 'loss: ' + str(update['result'][0]['update_id'] - a - 1 + '\n')
                print(loss)
                log_file.write(loss)

            log_file.close()
            a = update['result'][0]['update_id']
            b += 1

            chat_id = update['result'][0]['message']['chat']['id']
            chat_id = str(chat_id)


            if update['result'][0]['message']['text'][0] == '/':

                c = Library.run_command(update['result'][0]['message']['text'][1:], chat_id)
                update_id = update['result'][0]['update_id']
                if c == 0:
                    continue


            Library.send(chat_id, 'Напиши мне "/get", чтобы получить анекдот!')
            update_id = update['result'][0]['update_id']

    except KeyboardInterrupt:
        log_file = open('Log.txt', 'a')
        log_file.write('KeyboardInterrupt\n')
        break