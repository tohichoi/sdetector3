import datetime
import re
import time

# POSIX timestamp
# • Count of SI seconds since 1970-01-01 00:00:00 UTC
# • Except the leap seconds, which are “ignored”
# • Exactly 86400 seconds in a day
# “Always measure and store time in UTC or UNIX timestamps”
# • “If you are taking in user input that is in local time,
# immediately convert it to UTC. If that conversion
# would be ambiguous let the user know.”
# • “Rebase for Formatting (then throw away that filthy
# offset aware datetime object)”
# calendas
# datetime 과 무관
# 예외: Use calendar.timegm(tuple) to turn a struct_time
# (“time tuple”) in UTC into a POSIX timestamp
#
# datetime
# Naive, which have no timezone information
# • Aware, which have timezone information
# pytz
# • Update pytz regularly for changes in timezones
# seoul=pytz.timezone('Asia/Seoul')
# seoul.localize(datetime(2011, 11, 6, 5, 30))
# Out[11]: datetime.datetime(2011, 11, 6, 5, 30, tzinfo=<DstTzInfo 'Asia/Seoul' KST+9:00:00 STD>)
import pytz


class DateUtil:
    @staticmethod
    def get_now():
        zone = pytz.timezone('Asia/Seoul')
        return zone.localize(datetime.datetime.now())

    @staticmethod
    def get_current_datestring():
        return str(DateUtil.get_now().date())

    @staticmethod
    def get_current_timestring():
        s = DateUtil.get_now().replace(microsecond=0).isoformat()
        s = re.sub('[-:]', '', s)
        return s

    # 현재 시간 timestamp
    @staticmethod
    def get_posix_timestamp():
        return time.time()

    # if t==None, the current instant
    # else provided POSIX timestamp
    @staticmethod
    def get_posix_gmtime(t=None):
        return time.gmtime(None if t else DateUtil.get_posix_timestamp())

    # example:
    # 날짜는 없고 시간만 있음
    # localtime 이라고 가정함
    # st = datetime.time(hour=6, minute=30, second=0)
    # et = datetime.time(hour=2, minute=0, second=0)
    # today = datetime.date.today()
    #
    # t = datetime.datetime.combine(today, datetime.time(hour=7, minute=0, second=0))
    # in_range = is_time_in_range(st, et, t)  # True
    # print(in_range)
    #
    # t = datetime.datetime.combine(today, datetime.time(hour=23, minute=0, second=0))
    # in_range = is_time_in_range(st, et, t)  # True
    # print(in_range)
    #
    # t = datetime.datetime.combine(today, datetime.time(hour=1, minute=0, second=0))
    # in_range = is_time_in_range(st, et, t)  # False
    # print(in_range)
    @staticmethod
    def is_time_in_range(st, et, t=None):
        zone = pytz.timezone('Asia/Seoul')

        t = t if t else datetime.datetime.now()
        t_local = t.astimezone(zone)
        t_utc = t.astimezone(pytz.utc)

        today_local = zone.localize(t)

        st2 = datetime.datetime.combine(today_local, st)
        et2 = datetime.datetime.combine(today_local, et)

        st2_local = st2.astimezone(zone)
        et2_local = et2.astimezone(zone)

        st2_utc = st2.astimezone(pytz.utc)
        et2_utc = et2.astimezone(pytz.utc)

        if st2_utc.hour > et2_utc.hour:
            et2_utc = et2_utc + datetime.timedelta(days=1)
        if st2_local.hour > st2_local.hour:
            et2_local = et2_local + datetime.timedelta(days=1)

        # print('now')
        # print(f'local: {t_local.isoformat()}')
        # print(f'utc  : {t_utc.isoformat()}')
        #
        # print('st')
        # print(f'naive: {st2.isoformat()}')
        # print(f'local: {st2_local.isoformat()}')
        # print(f'utc  : {st2_utc.isoformat()}')
        #
        # print('et')
        # print(f'naive: {et2.isoformat()}')
        # print(f'local: {et2_local.isoformat()}')
        # print(f'utc  : {et2_utc.isoformat()}')

        return st2_utc <= t_utc <= et2_utc
