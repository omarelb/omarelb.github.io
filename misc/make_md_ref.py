import pyperclip
import sys

res = pyperclip.paste()
res = res.lower().replace(' ', '-')
pyperclip.copy(res)
