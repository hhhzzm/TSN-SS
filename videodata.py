import os

path = r"M:\Dataset\hmdb51_org"
for root, dirs, items in os.walk(path):
    for item in items:
        item2unrar = os.path.join(path, item)
        if item.split('.')[1] != 'rar':
            continue
        operation = "M:\\Dataset\\hmdb51_org\\UnRAR.exe e %s M:\\Dataset\\hmdb51_org\\videos" % (item2unrar)
        os.system(operation)
        print(item, "extracted")
