import logging
import os
from datetime import datetime

# Create logs directory if it does not exist
current_time = str(datetime.now().strftime('%Y-%m-%d-%I-%M-%S'))
log_file = current_time+".log"
log_path = os.path.join(os.getcwd(), 'logs', log_file)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Create a log file
logging.basicConfig(filename=log_path,
                    filemode='a',
                    level=logging.INFO,
                    format='time: ' + current_time + '\n'
                           'level name: %(levelname)s\n'
                           'path: %(pathname)s\n'
                           'line number: %(lineno)d\n'
                           'message: %(message)s\n\n')