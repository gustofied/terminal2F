import re


def skip_ws(s: str, i: int = 0):
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    
    return i, s[i:]

print(skip_ws("  ADADm adsad adasd.  sd"))
# def parse_json(json_string: str) -> dict | list:
#     return [1]


