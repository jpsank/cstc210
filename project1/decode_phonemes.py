import pronouncing
import random

def decode(text):
    phones_cache = {}
    for verse in text.split("\n\n"):
        for line in verse.split("\n"):
            out = ""
            left, right = line.split(), []
            while left or right:
                while left:
                    string = "^" + " ".join(left) + "$"
                    if string in phones_cache:
                        words = phones_cache[string]
                    else:
                        try:
                            words = pronouncing.search(string)
                        except Exception as e:
                            words = []
                    if words:
                        phones_cache[string] = words
                        break
                    right.append(left.pop())
                
                if not left:
                    out += "? "
                    left = right[1:]
                    right = []
                else:
                    word = random.choice(words)
                    out += word + " "
                    left = right
                    right = []
            yield out
        yield ""

if __name__ == '__main__':
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)

    for line in decode('\n'.join(contents)):
        print(line)