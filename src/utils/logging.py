import datetime
from .load_conf import load_conf


def log(category, message):
    cfg = load_conf()
    log_file = cfg["General"]["log_file"]
    chunk_file = cfg["General"]["chunk_file"]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if category == "QUERY_RESULTS":
        with open(chunk_file, 'a') as file:
            query_indicator= f"o ----------------------------------------------------------------------------------- \no\no[{timestamp}]\no\no ----------------------------------------------------------------------------------- \n"
            file.write(query_indicator)
            file.write(message)
            file.write("\n\n\n")
    else:
        log_entry = f"[{timestamp}] {category}: {message}\n"
        with open(log_file, 'a') as file:
            file.write(log_entry)

