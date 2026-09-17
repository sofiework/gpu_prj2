import sys, markdown, warnings
warnings.filterwarnings('ignore')
from weasyprint import HTML
src, out = sys.argv[1], sys.argv[2]
body = markdown.markdown(open(src).read(), extensions=['tables','fenced_code'])
css = """
@page { size: letter; margin: 0.9in 1in; }
body { font-family: "Liberation Serif","Times New Roman",serif; font-size: 11pt; line-height: 1.18; }
h1 { font-size: 14pt; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 10pt 0 4pt; }
h3,h4,h5,h6 { font-size: 11pt; margin: 8pt 0 3pt; }
p, li { margin: 0 0 5pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 7.5pt; margin: 4pt 0 8pt;
        font-family: "Liberation Sans","Arial",sans-serif; }
th, td { border: 0.5pt solid #999; padding: 1.5pt 3pt; text-align: right; }
th { background: #eee; font-weight: bold; }
td:first-child, th:first-child { text-align: left; }
pre { font-family: "DejaVu Sans Mono",monospace; font-size: 7pt; background: #f4f4f4;
      padding: 3pt; white-space: pre-wrap; border-left: 2pt solid #ccc; margin: 3pt 0; }
code { font-family: "DejaVu Sans Mono",monospace; font-size: 8.5pt; }
"""
doc = HTML(string=f"<style>{css}</style>{body}").render()
doc.write_pdf(out)
print(f"{out}: {len(doc.pages)} pages")
