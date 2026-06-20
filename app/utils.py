import logging
from logging.handlers import RotatingFileHandler
import re
import os
from app.config import settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Rotating file handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")
        
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def clean_text(text: str) -> str:
    """
    Clean extracted text.
    - Remove excessive whitespace and multiple newlines
    - Normalize space and line-endings
    - Keep general structure but clean garbage characters
    """
    if not text:
        return ""
    # Replace page-break markers (\x0c or \f)
    text = text.replace("\x0c", " ").replace("\f", " ")
    
    # Replace carriage returns
    text = text.replace("\r", "\n")
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace more than two newlines with exactly two newlines (to preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing spaces from each line
    lines = [line.strip() for line in text.split("\n")]
    
    # Join with newlines and clean again
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def format_subject_name(filename: str) -> str:
    """Converts a filename like '11-radio-navigation-2014.pdf' to 'Radio Navigation'"""
    if not filename:
        return "Unknown Subject"
    name = filename.lower().replace(".pdf", "")
    name = re.sub(r'^\d+-?', '', name)
    name = re.sub(r'-\d{4}$', '', name)
    name = name.replace("-", " ").replace("_", " ")
    if name == "instruments":
        name = "instrumentation"
    return name.title().strip()
