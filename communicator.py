# encoding: utf-8

import glob
import subprocess
import mydateutil
from fileutil import create_logging, VideoFileWritingThread, FileUtil
import logging
from telegram import ChatAction, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Updater, CommandHandler
import json
from motiondetector import TelegramData
from telegram import Bot
import os
import pyscreenshot as ImageGrab
from threading import Timer


def find_last_logfile():
    filelist=glob.glob('log/*.txt')
    if len(filelist) < 1:
        return None

    filelist2 = sorted(filelist, key=os.path.getmtime, reverse=True)

    # ftime = []
    # for f in os.listdir('log'):
    #     fpath = 'log/'+f
    #     tm = os.path.getmtime(fpath)
    #     ftime.append((fpath, tm))
    # ftime2 = sorted(ftime, key=lambda x : x[1], reverse=True)

    return filelist2[0]


def __delete_file(args):

    os.unlink(args)


def __kill():

    try:
        subprocess.check_call(['pkill', '-9', '-f', 'motiondetector.py'])
        msg = '종료 성공'
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            msg = f'순탐이가 실행중이지 않음'
        else:
            msg = f'종료 오류:\n{e.args[1]} returned {e.returncode}'

    return msg


def get_help_string():

    msg = ''
    msg += '/status : 동작 상태 보기\n'
    msg += '/restart : 순탐이 재시작\n'
    msg += '/screen : 스크린샷 보기\n'
    msg += '/kill : 순탐이 종료\n'

    return msg


def help_(update, context):

    msg = get_help_string()

    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


def screen(update, context):

    im = ImageGrab.grab()

    d = mydateutil.DateUtil.get_current_timestring()
    fname = f'log/{d}-screen.png'
    im.save(fname)

    context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(fname, 'rb'))

    Timer(5.0, lambda del_f: os.remove(del_f), args=[fname]).start()


def kill(update, context):

    msg = __kill()

    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


def restart(update, context):

    msg = __kill()
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    try:
        p = subprocess.Popen(['bash', 'run_live.sh'])
        msg = '재실행 성공'
    except subprocess.CalledProcessError as e:
        msg = '재실행 오류:\n' + str(e.args)

    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


def start(update, context):

    msg = get_help_string()
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


def status(update, context):
    last_logfile = find_last_logfile()
    if last_logfile is None:
        s = '로그파일이 없는데? 순탐이가 실행중인가?'
    else:
        s = subprocess.check_output(['tail', last_logfile]).decode("utf-8")

    context.bot.send_message(chat_id=update.effective_chat.id, text=s)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        #    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s : %(funcName)s : %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if os.path.exists('config/telegram.json'):
        with open('config/telegram.json') as fd:
            cf = json.load(fd)
            TelegramData.CHAT_ID = cf['bot_chatid']
            TelegramData.TOKEN = cf['bot_token']
            TelegramData.bot = Bot(TelegramData.TOKEN)

    updater = Updater(token=TelegramData.TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    help_handler = CommandHandler('help', help_)
    dispatcher.add_handler(help_handler)

    status_handler = CommandHandler('status', status)
    dispatcher.add_handler(status_handler)

    restart_handler = CommandHandler('restart', restart)
    dispatcher.add_handler(restart_handler)

    kill_handler = CommandHandler('kill', kill)
    dispatcher.add_handler(kill_handler)

    screen_handler = CommandHandler('screen', screen)
    dispatcher.add_handler(screen_handler)

    logging.info("Starting")

    updater.start_polling()
    updater.idle()

    logging.info("Finishing")