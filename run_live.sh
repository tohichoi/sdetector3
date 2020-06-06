py ./motiondetector.py `cat config/address.txt` |& tee log/$(date --iso-8601=seconds).txt
