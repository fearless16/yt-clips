"""
utils/logger.py — Premium ANSI Colored Logging for a Professional Console Experience.
"""
import logging
import sys
from pathlib import Path

# ANSI Color Codes
BLUE    = "\033[94m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

class PremiumFormatter(logging.Formatter):
    """Custom Formatter with Premium Look & Symbols"""
    
    LEVEL_MAP = {
        logging.DEBUG:    (f"{MAGENTA}⚙{RESET}", MAGENTA),
        logging.INFO:     (f"{BLUE}ℹ{RESET}",    BLUE),
        logging.WARNING:  (f"{YELLOW}⚠️{RESET}",  YELLOW),
        logging.ERROR:    (f"{RED}❌{RESET}",   RED),
        logging.CRITICAL: (f"{RED}{BOLD}🔥{RESET}", RED),
    }

    def format(self, record):
        symbol, color = self.LEVEL_MAP.get(record.levelno, (f"{BLUE}•{RESET}", BLUE))
        
        # Success message special handling
        msg = record.getMessage()
        is_success = any(keyword in msg.lower() for keyword in ["complete", "success", "done", "uploaded", "saved"])
        
        if is_success:
            symbol = f"{GREEN}✅{RESET}"
            msg = f"{GREEN}{BOLD}{msg}{RESET}"
        
        time_str = self.formatTime(record, "%H:%M:%S")
        
        # Clean formatting
        return f"[{color}{time_str}{RESET}] {symbol} {BOLD}{record.name:<10}{RESET} {msg}"

def get_logger(name: str, log_file: str = "logs/pipeline.log", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(PremiumFormatter())
    logger.addHandler(c_handler)
    
    # File Handler (Plain text for files)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    f_handler = logging.FileHandler(log_file)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    
    return logger
