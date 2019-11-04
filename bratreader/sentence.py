from bratreader.word import Word
import re


class Sentence(object):

    def __init__(self, key, line, start):
        """
        Sentence object.

        :param key: The key to which this sentence belongs.
        :param line: The line on which this sentences occurs.
        :param start: The start index of this line in characters.
        """
        self.key = key
        self.words = []
        self.start = start
        self.end = start + len(line)
        
        sent_start = start
        ## There are double spaces
        wlist = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\S+', line)]
        for windex, wobj in enumerate(wlist):
            w, start, end = wobj
            start = start + sent_start
            end = end + sent_start
            self.words.append(Word(key=windex,
                                   sentkey=self.key,
                                   form=w,
                                   start=start,
                                   end=end))

    def getwordsinspan(self, start, end):
        """
        Retrieve all words in the specified character span.

        :param start: The start index in characters.
        :param end: The end index in characters.
        :return a list of words that fall inside the span.
        """
        return [word for word in self.words if
                (word.start <= start < word.end)
                or (word.start < end <= word.end)
                or (start < word.start < end and start < word.end < end)]
