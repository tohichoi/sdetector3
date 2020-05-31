import datetime
import re


class DateUtil:
    @staticmethod
    def get_current_timestring():
        s = datetime.datetime.now().replace(microsecond=0).isoformat()
        s = re.sub('[-:]', '', s)
        return s