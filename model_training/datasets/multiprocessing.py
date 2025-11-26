import multiprocessing as mp
import logging

class SimpleFileLogger:
    """Redirect writes to a plain text file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, "w", buffering=1)  # line-buffered

    def write(self, text):
        # write line by line
        for line in text.rstrip().splitlines():
            self.file.write(f"{line}\n")

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass 

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# def preprocess_large_dataset(Telescope_directory):
#     name = os.path.basename(Telescope_directory)
#     # logger = setup_logger(name, (os.path.join("/data/Dataset_Compilation_and_Statistics/logs",f'preprocess-{name}.log')))
#     # sys.stdout = StreamToLogger(logger)

#     # Create logger object
#     file_logger = SimpleFileLogger(os.path.join("/data/Dataset_Compilation_and_Statistics/logs",f'preprocess-{name}.log'))
#     sys.stdout = file_logger
#     sys.stderr = file_logger

#     fail_directory =Telescope_directory +"-Errors"
#     os.makedirs(fail_directory, exist_ok=True)