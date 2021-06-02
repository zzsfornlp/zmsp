#

# -----
# for data
"""
- Generally two layers: instance, field
-- instance can be automatically recursive (but get specific classes for common ones, like "Sentence", "Paragraph")
-- field must specify details by themselves (usually pieces of instances, like "word_seq", "pos_seq")
"""

from .base import BaseDataItem
from .field import DataField, SeqField, NpArrField, DepTreeField
from .general import GeneralInstance, GeneralSentence, GeneralParagraph, GeneralDocument
