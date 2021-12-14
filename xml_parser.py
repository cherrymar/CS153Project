import xml.etree.ElementTree as ET

'''
input: filename of xml file
output: extracts the annotations from the xml file
'''
def getAnnotations(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    annotations =[]
    for child in root:
        if child.tag == "object":
            annotations.append(child[0].text)

    return annotations