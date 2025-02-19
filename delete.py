import xml.etree.ElementTree as ET

# XML 파일 로드
tree = ET.parse('annotations.xml')
root = tree.getroot()

# 모든 이미지 요소에서 box 태그 제거
for image in root.findall('.//image'):
    boxes = image.findall('box')
    for box in boxes:
        image.remove(box)

# 수정된 XML 저장 (들여쓰기 적용)
ET.indent(tree, space="  ", level=0)
tree.write('cleaned_annotations.xml', 
          encoding='utf-8', 
          xml_declaration=True,
          short_empty_elements=False)
