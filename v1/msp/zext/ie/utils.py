# =====
# other helpers

# split by Camel Capital letter or "-"/".", return list of lowercased parts
def split_camel(ss):
    prev_upper = False
    all_cs = []
    for c in ss:
        if str.isupper(c):
            if not prev_upper:  # splitting point
                all_cs.append("-")
                prev_upper = True
        elif str.islower(c):
            prev_upper = False
        else:
            prev_upper = False
            if c == ".":
                c = '-'
            elif c == "/":
                pass
            assert c=="-" or c=="/", f"Illegal char {c} for label"
        all_cs.append(str.lower(c))  # here, first lower all
    ret = [z for z in "".join(all_cs).split("-") if len(z)>0]
    return ret

# norm things into camel format without "-"
def rearrange_camel(ss):
    parts = split_camel(ss)
    return "".join([str.upper(z[0])+z[1:] for z in parts])

# todo(note): specific to ACE/ERE style
# mainly for Entity types
LABEL2LEX_MAP = {
    "per": "person", "org": "organization", "gpe": "geopolitics", "loc": "location", "fac": "facility",
    "veh": "vehicle", "wea": "weapon",
}
